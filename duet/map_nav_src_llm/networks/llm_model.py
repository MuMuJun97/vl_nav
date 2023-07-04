import numpy as np
import collections
from typing import Dict, Iterable, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel
import json
import logging
import math
import os
import sys
from io import open
from typing import Callable, List, Tuple
import numpy as np
import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor, device, dtype

from transformers import BertPreTrainedModel
from pathlib import Path
from .ops import create_transformer_encoder
from .ops import extend_neg_masks, gen_seq_masks, pad_tensors_wgrad
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
BertLayerNorm = torch.nn.LayerNorm


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


def llama_model_in_debug_model(lang_encoder_path):
    from transformers.models.llama import LlamaForCausalLM
    from transformers.utils import ContextManagers
    from transformers.modeling_utils import no_init_weights
    from transformers.generation import GenerationConfig
    # Instantiate model.
    init_contexts = [no_init_weights(_enable=True)]
    import pickle
    from pathlib import Path
    with open(lang_encoder_path, "rb") as f:
        config = pickle.load(f)
        config.intermediate_size = 1024
        config.num_hidden_layers = 2
        config.hidden_size = 1024
    model_args = ()
    model_kwargs = {}
    with ContextManagers(init_contexts):
        model = LlamaForCausalLM(config, *model_args, **model_kwargs)
    model.is_loaded_in_8bit = False
    # make sure token embedding weights are still tied if needed
    model.tie_weights()
    # Set model in evaluation mode to deactivate DropOut modules by default
    model.eval()
    # If it is a model with generation capabilities, attempt to load the generation config
    if model.can_generate():
        pretrained_model_name_or_path = Path(lang_encoder_path).parent.resolve().__str__()
        try:
            kwargs = {}
            model.generation_config = GenerationConfig.from_pretrained(
                pretrained_model_name_or_path,
                cache_dir=None,
                force_download=False,
                resume_download=False,
                proxies=None,
                local_files_only=False,
                use_auth_token=None,
                revision=None,
                subfolder='',
                _from_auto=False,
                _from_pipeline=None,
                **kwargs,
            )
        except Exception as e:
            print(e)
    return model


class LangModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # LLaMa-7B, 'bigscience/bloom-560m',
        self.tokenizer_path = config.tokenizer_path
        if 'bloom' in self.tokenizer_path:
            self.is_bloom = True
        else:
            self.is_bloom = False

        local_files_only = False

        if config.precision == 'fp16':
            self.model_type = torch.float16
        elif 'bf16' in config.precision or 'bfloat16' in config.precision:
            self.model_type = torch.bfloat16
        else:
            self.model_type = torch.float32

        print("************ Use dtype: {} ************\n".format(self.model_type))

        if 'model_config.pkl' in self.tokenizer_path:
            # use local llama-7b
            _tokenizer_path = Path(self.tokenizer_path).parent.resolve().__str__()
            self.tokenizer = AutoTokenizer.from_pretrained(
                _tokenizer_path, local_files_only=local_files_only
            )
            is_local_llama = True
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path, local_files_only=local_files_only
            )
            is_local_llama = False

        self.cand_token = ['<cand>']
        self.his_token = ['<hist>']
        self.exec_token = ['<exec{}>'.format(i) for i in range(100)]
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.cand_token + self.his_token + self.exec_token}
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})

        self.cand_token_id = self.tokenizer.encode("<cand>", add_special_tokens=False)
        self.hist_token_id = self.tokenizer.encode("<hist>", add_special_tokens=False)

        if is_local_llama:
            self.lang_model = llama_model_in_debug_model(self.tokenizer_path)
            self.lang_model = self.lang_model.to(self.model_type)
        else:
            if self.tokenizer_path == "facebook/opt-iml-1.3b":
                self.lang_model = AutoModelForCausalLM.from_pretrained(
                    self.tokenizer_path, local_files_only=local_files_only
                ).to(self.model_type)  # bfloat16, float16, float32

                ### [+] re-init opt-iml weights
                # from transformers import OPTForCausalLM, AutoConfig
                # opt_config = AutoConfig.from_pretrained(self.tokenizer_path)
                # self.lang_model = OPTForCausalLM(opt_config).to(self.model_type)
                # model_size = sum(t.numel() for t in self.lang_model.parameters())
                # print(f"OPT-IML size: {model_size / 1000 ** 2:.1f}M parameters")
            else:
                self.lang_model = AutoModelForCausalLM.from_pretrained(
                    self.tokenizer_path, local_files_only=local_files_only
                ).to(self.model_type)  # bfloat16, float16, float32

        # llama-7b dim=4096, bloom dim=1024,
        self.hidden_size = self.lang_model.config.hidden_size
        self.lang_model.resize_token_embeddings(len(self.tokenizer))

        self.mapper = nn.Sequential(
            nn.Linear(768, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

    def tokenize(self, text):
        batch_text = self.tokenizer(
            text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False
        )
        return batch_text

    def forward(self, *input, **kwargs):
        input_ids = kwargs["input_ids"] if "input_ids" in kwargs else input[0]  # B,L

        hist_locations = (input_ids >= self.hist_token_id[0]) & (input_ids <= self.hist_token_id[-1])
        cand_locations = (input_ids >= self.cand_token_id[0]) & (input_ids <= self.cand_token_id[-1])

        if 'opt' in self.tokenizer_path:
            inputs_embeds = self.lang_model.model.decoder.embed_tokens(input_ids)
        else:
            if not self.is_bloom:
                # llama-7b
                inputs_embeds = self.lang_model.model.embed_tokens(input_ids)
            else:
                # bloom-560M
                inputs_embeds = self.lang_model.transformer.word_embeddings(input_ids)

        if cand_locations.sum() != 0:
            inputs_embeds[cand_locations] += self.mapper(kwargs['cand_vis'])
        if hist_locations.sum() != 0:
            inputs_embeds[hist_locations] += self.mapper(kwargs['hist_vis'])

        if 'cand_vis' in kwargs:
            kwargs.pop('cand_vis')
        if 'hist_vis' in kwargs:
            kwargs.pop('hist_vis')

        kwargs["input_ids"] = None
        kwargs["inputs_embeds"] = inputs_embeds

        labels = None
        if 'labels' in kwargs:
            labels = kwargs.pop('labels')

        if not self.is_bloom:
            outputs = self.lang_model.model(*input, **kwargs)
        else:
            outputs = self.lang_model.transformer(*input, **kwargs)

        hidden_states = outputs[0]
        logits = self.lang_model.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.lang_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return loss, logits, hidden_states[cand_locations]


class ImageEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.loc_linear = nn.Linear(config.angle_feat_size + 3, config.hidden_size)
        self.loc_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        if config.obj_feat_size > 0 and config.obj_feat_size != config.image_feat_size:
            self.obj_linear = nn.Linear(config.obj_feat_size, config.hidden_size)
            self.obj_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        else:
            self.obj_linear = self.obj_layer_norm = None

        # 0: non-navigable, 1: navigable, 2: object
        self.nav_type_embedding = nn.Embedding(3, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if config.num_pano_layers > 0:
            self.pano_encoder = create_transformer_encoder(
                config, config.num_pano_layers, norm=True
            )
        else:
            self.pano_encoder = None

    def forward(
            self, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, type_embed_layer
    ):
        device = traj_view_img_fts.device
        has_obj = traj_obj_img_fts is not None

        traj_view_img_embeds = self.img_layer_norm(self.img_linear(traj_view_img_fts))
        if has_obj:
            if self.obj_linear is None:
                traj_obj_img_embeds = self.img_layer_norm(self.img_linear(traj_obj_img_fts))
            else:
                raise NotImplementedError
                # traj_obj_img_embeds = self.obj_layer_norm(self.obj_linear(traj_obj_img_embeds))
            traj_img_embeds = []
            for view_embed, obj_embed, view_len, obj_len in zip(
                    traj_view_img_embeds, traj_obj_img_embeds, traj_vp_view_lens, traj_vp_obj_lens
            ):
                if obj_len > 0:
                    traj_img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0))
                else:
                    traj_img_embeds.append(view_embed[:view_len])
            traj_img_embeds = pad_tensors_wgrad(traj_img_embeds)
            traj_vp_lens = traj_vp_view_lens + traj_vp_obj_lens
        else:
            traj_img_embeds = traj_view_img_embeds
            traj_vp_lens = traj_vp_view_lens

        traj_embeds = traj_img_embeds + \
                      self.loc_layer_norm(self.loc_linear(traj_loc_fts)) + \
                      self.nav_type_embedding(traj_nav_types) + \
                      type_embed_layer(torch.ones(1, 1).long().to(device))
        traj_embeds = self.layer_norm(traj_embeds)
        traj_embeds = self.dropout(traj_embeds)

        traj_masks = gen_seq_masks(traj_vp_lens)
        if self.pano_encoder is not None:
            traj_embeds = self.pano_encoder(
                traj_embeds, src_key_padding_mask=traj_masks.logical_not()
            )

        split_traj_embeds = torch.split(traj_embeds, traj_step_lens, 0)
        split_traj_vp_lens = torch.split(traj_vp_lens, traj_step_lens, 0)
        return split_traj_embeds, split_traj_vp_lens


class ClsPrediction(nn.Module):
    def __init__(self, hidden_size, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        """
        hidden_states: (N, L_{hidden}, D)
        attention_mask: (N, H, L_{hidden}, L_{hidden})
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # recurrent vlnbert use attention scores
        outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GraphLXRTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Visn self-att and FFN layer
        self.visn_self_att = BertAttention(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

    def forward(
            self, visn_feats, visn_attention_mask,
            graph_sprels=None
    ):
        if graph_sprels is not None:
            visn_attention_mask = visn_attention_mask + graph_sprels
        visn_att_output = self.visn_self_att(visn_feats, visn_attention_mask)[0]

        visn_inter_output = self.visn_inter(visn_att_output)
        visn_output = self.visn_output(visn_inter_output, visn_att_output)

        return visn_output

    def forward_lang2visn(
            self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,
    ):
        lang_att_output = self.visual_attention(
            lang_feats, visn_feats, ctx_att_mask=visn_attention_mask
        )[0]
        lang_att_output = self.lang_self_att(
            lang_att_output, lang_attention_mask
        )[0]
        lang_inter_output = self.lang_inter(lang_att_output)
        lang_output = self.lang_output(lang_inter_output, lang_att_output)
        return lang_output


class CrossmodalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_x_layers = config.num_x_layers
        self.x_layers = nn.ModuleList(
            [GraphLXRTXLayer(config) for _ in range(self.num_x_layers)]
        )

    # txt_embeds[B, L, 768], img_embeds[B, N(valid views), 768], graph_sprels[B, 1, N, N] spatial relationship
    def forward(self, img_embeds, img_masks, graph_sprels=None):
        extended_img_masks = extend_neg_masks(img_masks)  # (N, 1(H), 1(L_q), L_v)
        for layer_module in self.x_layers:
            img_embeds = layer_module(
                img_embeds, extended_img_masks,
                graph_sprels=graph_sprels
            )
        return img_embeds  # -> img_embeds[B, N(valid views), 768]


class LocalVPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vp_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size * 2 + 6, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.encoder = CrossmodalEncoder(config)

    def vp_input_embedding(self, split_traj_embeds, split_traj_vp_lens, vp_pos_fts):
        vp_img_embeds = pad_tensors_wgrad([x[-1] for x in split_traj_embeds])
        vp_lens = torch.stack([x[-1] + 1 for x in split_traj_vp_lens], 0)
        vp_masks = gen_seq_masks(vp_lens)
        max_vp_len = max(vp_lens)

        batch_size, _, hidden_size = vp_img_embeds.size()
        device = vp_img_embeds.device
        # add [stop] token at beginning
        vp_img_embeds = torch.cat(
            [torch.zeros(batch_size, 1, hidden_size).to(device), vp_img_embeds], 1
        )[:, :max_vp_len]
        vp_embeds = vp_img_embeds + self.vp_pos_embeddings(vp_pos_fts)

        return vp_embeds, vp_masks

    def forward(
            self, txt_embeds, txt_masks, split_traj_embeds, split_traj_vp_lens, vp_pos_fts
    ):
        vp_embeds, vp_masks = self.vp_input_embedding(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )
        vp_embeds = self.encoder(txt_embeds, txt_masks, vp_embeds, vp_masks)
        return vp_embeds


class GlobalMapEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gmap_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.gmap_step_embeddings = nn.Embedding(config.max_action_steps, config.hidden_size)
        self.encoder = CrossmodalEncoder(config)

        if config.graph_sprels:
            self.sprel_linear = nn.Linear(1, 1)
        else:
            self.sprel_linear = None

    def _aggregate_gmap_features(
            self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
    ):
        batch_size = len(split_traj_embeds)
        device = split_traj_embeds[0].device

        batch_gmap_img_fts = []
        for i in range(batch_size):
            visited_vp_fts, unvisited_vp_fts = {}, {}
            vp_masks = gen_seq_masks(split_traj_vp_lens[i])
            max_vp_len = max(split_traj_vp_lens[i])
            i_traj_embeds = split_traj_embeds[i][:, :max_vp_len] * vp_masks.unsqueeze(2)
            for t in range(len(split_traj_embeds[i])):
                visited_vp_fts[traj_vpids[i][t]] = torch.sum(i_traj_embeds[t], 0) / split_traj_vp_lens[i][t]
                for j, vp in enumerate(traj_cand_vpids[i][t]):
                    if vp not in visited_vp_fts:
                        unvisited_vp_fts.setdefault(vp, [])
                        unvisited_vp_fts[vp].append(i_traj_embeds[t][j])

            gmap_img_fts = []
            for vp in gmap_vpids[i][1:]:
                if vp in visited_vp_fts:
                    gmap_img_fts.append(visited_vp_fts[vp])
                else:
                    gmap_img_fts.append(torch.mean(torch.stack(unvisited_vp_fts[vp], 0), 0))
            gmap_img_fts = torch.stack(gmap_img_fts, 0)
            batch_gmap_img_fts.append(gmap_img_fts)

        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        # add a [stop] token at beginning
        batch_gmap_img_fts = torch.cat(
            [torch.zeros(batch_size, 1, batch_gmap_img_fts.size(2)).to(device), batch_gmap_img_fts],
            dim=1
        )
        return batch_gmap_img_fts

    def gmap_input_embedding(
            self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens
    ):
        gmap_img_fts = self._aggregate_gmap_features(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
        )
        gmap_embeds = gmap_img_fts + \
                      self.gmap_step_embeddings(gmap_step_ids) + \
                      self.gmap_pos_embeddings(gmap_pos_fts)
        gmap_masks = gen_seq_masks(gmap_lens)
        return gmap_embeds, gmap_masks

    def forward(
            self, txt_embeds, txt_masks,
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens, graph_sprels=None
    ):
        gmap_embeds, gmap_masks = self.gmap_input_embedding(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens
        )

        if self.sprel_linear is not None:
            graph_sprels = self.sprel_linear(graph_sprels.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None

        gmap_embeds = self.encoder(
            txt_embeds, txt_masks, gmap_embeds, gmap_masks,
            graph_sprels=graph_sprels
        )
        return gmap_embeds


class FuseEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size, eps=1e-12),
            # nn.Linear(hidden_size, hidden_size),
            # nn.LayerNorm(hidden_size, eps=1e-12)
        )

    def forward(self, x):
        return self.encoder(x)


class GlocalTextPathNavCMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.img_embeddings = ImageEmbeddings(config)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.lang_model = LangModel(config)
        self.hidden_size = self.lang_model.hidden_size
        self.model_type = self.lang_model.model_type

        self.local_encoder = LocalVPEncoder(config)
        self.global_encoder = GlobalMapEncoder(config)
        self.sap_fuse_linear = ClsPrediction(config.hidden_size, input_size=config.hidden_size * 2)

        self.eqa_ans_dim = 35
        self.eqa_embeddings = nn.Embedding(self.eqa_ans_dim, config.hidden_size)

        self.fuse_encoder = FuseEncoder(config.hidden_size)

        self.out_head = ClsPrediction(self.lang_model.hidden_size).to(self.lang_model.model_type)

        self.instruction = None
        self.history = None
        self.hist_vis = None

        self.init_weights()

    def forward_panorama_per_step(
            self, view_img_fts, obj_img_fts, loc_fts, nav_types, view_lens, obj_lens
    ):
        device = view_img_fts.device
        has_obj = obj_img_fts is not None

        view_img_embeds = self.img_embeddings.img_layer_norm(
            self.img_embeddings.img_linear(view_img_fts)
        )
        if has_obj:
            if self.img_embeddings.obj_linear is None:
                obj_img_embeds = self.img_embeddings.img_layer_norm(
                    self.img_embeddings.img_linear(obj_img_fts)
                )
            else:
                obj_img_embeds = self.img_embeddings.obj_layer_norm(
                    self.img_embeddings.obj_linear(obj_img_fts)
                )
            img_embeds = []
            for view_embed, obj_embed, view_len, obj_len in zip(
                    view_img_embeds, obj_img_embeds, view_lens, obj_lens
            ):
                if obj_len > 0:
                    img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0))
                else:
                    img_embeds.append(view_embed[:view_len])
            img_embeds = pad_tensors_wgrad(img_embeds)
            pano_lens = view_lens + obj_lens
        else:
            img_embeds = view_img_embeds
            pano_lens = view_lens

        pano_embeds = img_embeds + \
                      self.img_embeddings.loc_layer_norm(self.img_embeddings.loc_linear(loc_fts)) + \
                      self.img_embeddings.nav_type_embedding(nav_types) + \
                      self.token_type_embeddings(torch.ones(1, 1).long().to(device))

        pano_embeds = self.img_embeddings.layer_norm(pano_embeds)
        pano_embeds = self.img_embeddings.dropout(pano_embeds)

        pano_masks = gen_seq_masks(pano_lens, max_len=pano_embeds.shape[-2])
        if self.img_embeddings.pano_encoder is not None:
            pano_embeds = self.img_embeddings.pano_encoder(
                pano_embeds, src_key_padding_mask=pano_masks.logical_not()
            )
        return pano_embeds, pano_masks

    def forward(self, mode, batch, **kwargs):
        if mode == 'language':
            return

        elif mode == 'panorama':
            pano_embeds, pano_masks = self.forward_panorama_per_step(
                batch['view_img_fts'], batch['obj_img_fts'], batch['loc_fts'],
                batch['nav_types'], batch['view_lens'], batch['obj_lens']
            )
            pano_embeds.masked_fill_(pano_masks.logical_not().unsqueeze(-1), 0.)
            return pano_embeds, pano_masks

        elif mode == 'navigation':
            data_type = batch['data_type']
            vp_img_embeds = batch['vp_img_embeds']
            batch_size = vp_img_embeds.size(0)
            gmap_img_embeds, gmap_step_ids, gmap_pos_fts, \
                gmap_masks, gmap_pair_dists, gmap_visited_masks, gmap_vpids \
                = batch['gmap_img_embeds'], batch['gmap_step_ids'], batch['gmap_pos_fts'], \
                batch['gmap_masks'], batch['gmap_pair_dists'], batch['gmap_visited_masks'], batch['gmap_vpids'],

            # global branch [B, Nums, D=768]
            gmap_embeds = torch.zeros_like(gmap_img_embeds)
            for b_ix in range(len(data_type)):
                if data_type[b_ix] == 'eqa':
                    gmap_embeds[b_ix:b_ix+1] = gmap_img_embeds[b_ix:b_ix+1]
                else:
                    gmap_embeds[b_ix:b_ix+1] = gmap_img_embeds[b_ix:b_ix+1] + \
                                  self.global_encoder.gmap_step_embeddings(gmap_step_ids[b_ix:b_ix+1]) + \
                                  self.global_encoder.gmap_pos_embeddings(gmap_pos_fts[b_ix:b_ix+1])

            if self.global_encoder.sprel_linear is not None:
                graph_sprels = self.global_encoder.sprel_linear(
                    gmap_pair_dists.unsqueeze(3)).squeeze(3).unsqueeze(1)
            else:
                graph_sprels = None

            gmap_embeds = self.global_encoder.encoder(
                gmap_embeds, gmap_masks,
                graph_sprels=graph_sprels
            )

            ##### local branch #####
            vp_img_embeds, vp_pos_fts, vp_masks, \
                vp_nav_masks, vp_obj_masks, vp_cand_vpids = \
                batch['vp_img_embeds'], batch['vp_pos_fts'], batch['vp_masks'], \
                    batch['vp_nav_masks'], batch['vp_obj_masks'], batch['vp_cand_vpids']

            vp_embeds = torch.zeros_like(vp_img_embeds)
            for b_ix in range(len(data_type)):
                if data_type[b_ix] == 'eqa':
                    vp_embeds[b_ix:b_ix+1] = vp_img_embeds[b_ix:b_ix+1]
                else:
                    vp_embeds[b_ix:b_ix+1] = vp_img_embeds[b_ix:b_ix+1] \
                        + self.local_encoder.vp_pos_embeddings(vp_pos_fts[b_ix:b_ix+1])
            vp_embeds = self.local_encoder.encoder(vp_embeds, vp_masks)

            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))

            ##### fuse embeds #####
            global_gmap_embeds = gmap_embeds * fuse_weights.unsqueeze(-1)
            global_gmap_embeds.masked_fill_(gmap_visited_masks.unsqueeze(-1), 0.)
            global_gmap_embeds.masked_fill_(gmap_masks.logical_not().unsqueeze(-1), 0.)

            local_vp_embeds = vp_embeds * (1 - fuse_weights).unsqueeze(-1)
            local_vp_embeds.masked_fill_(vp_nav_masks.logical_not().unsqueeze(-1), 0.)

            fuse_embeds = torch.clone(global_gmap_embeds)
            fuse_embeds[:, 0] += local_vp_embeds[:, 0]  # stop

            for i in range(batch_size):
                if data_type[i] == 'eqa':
                    # fuse_embeds[i, 0] = torch.zeros_like(fuse_embeds[i, 0])
                    for j, vp in enumerate(gmap_vpids[i]):
                        if j > 0:
                            tmp = []
                            for l, _ in enumerate(vp_cand_vpids[i]):
                                if l > 0:
                                    tmp.append(local_vp_embeds[i, l])
                            fuse_embeds[i, j] = torch.stack(tmp, dim=0).mean(dim=0)
                else:
                    visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
                    tmp = {}
                    bw_logits = 0
                    for j, cand_vpid in enumerate(vp_cand_vpids[i]):
                        if j > 0:
                            if cand_vpid in visited_nodes:
                                bw_logits += local_vp_embeds[i, j]
                            else:
                                tmp[cand_vpid] = local_vp_embeds[i, j]
                    for j, vp in enumerate(gmap_vpids[i]):
                        if j > 0 and vp not in visited_nodes:
                            if vp in tmp:
                                fuse_embeds[i, j] += tmp[vp]
                            else:
                                fuse_embeds[i, j] += bw_logits

            fuse_embeds = self.fuse_encoder(fuse_embeds)
            fuse_embeds.masked_fill_(gmap_visited_masks.unsqueeze(-1), 0.)
            fuse_embeds.masked_fill_(gmap_masks.logical_not().unsqueeze(-1), 0.)

            ########### cross-modal #############
            if 'eqa' in data_type:
                max_len = max(self.eqa_ans_dim, gmap_masks.shape[-1])
                pad_gmap_masks = torch.zeros((gmap_masks.shape[0], max_len), device=gmap_masks.device).bool()
                pad_fuse_embeds = torch.zeros((fuse_embeds.shape[0], max_len, fuse_embeds.shape[-1]),
                    device=fuse_embeds.device, dtype=fuse_embeds.dtype)
                pad_gmap_visited_masks = torch.zeros((gmap_visited_masks.shape[0], max_len),
                                                     device=gmap_visited_masks.device).bool()
                pad_gmap_visited_masks[:, :gmap_visited_masks.shape[-1]] = gmap_visited_masks
                for idx in range(len(data_type)):
                    if data_type[idx] == 'eqa':
                        pad_gmap_masks[idx, :self.eqa_ans_dim] = torch.ones_like(pad_gmap_masks[idx, :self.eqa_ans_dim]).bool()
                        pad_fuse_embeds[idx, :self.eqa_ans_dim] = torch.cat([fuse_embeds[idx, 1:2]]*self.eqa_ans_dim) + \
                            self.eqa_embeddings(torch.arange(self.eqa_ans_dim).to(fuse_embeds.device))
                    else:
                        pad_gmap_masks[idx, :gmap_masks[idx].shape[-1]] = gmap_masks[idx]
                        pad_fuse_embeds[idx, :fuse_embeds.shape[-2]] = fuse_embeds[idx, :fuse_embeds.shape[-2]]

                gmap_masks, fuse_embeds = pad_gmap_masks, pad_fuse_embeds
                gmap_visited_masks = pad_gmap_visited_masks

            cand_masks = torch.clone(gmap_masks)
            cand_nums = cand_masks.sum(dim=-1)
            instruction = batch['instruction']
            history = batch['history']
            hist_vis = batch['hist_vis']
            hist_vis_input = []
            for vis in hist_vis:
                hist_vis_input.extend(vis)
            if hist_vis_input != []:
                hist_vis_input = torch.stack(hist_vis_input, dim=0)
            else:
                hist_vis_input = None

            cand_text = []
            for idx, cand_num in enumerate(cand_nums):
                if data_type[idx] == 'eqa':
                    cand_text.append(''.join(['<cand>'.format(i) for i in range(self.eqa_ans_dim)]))
                    cand_nums[idx] = self.eqa_ans_dim
                else:
                    cand_text.append(''.join(['<cand>'.format(i) for i in range(cand_num)]))
            cand_text = ['\nEnvironment: ' + text + '\nAgent: <s>' for text in cand_text]

            all_text = ["Conmander: {} \nHistory: {} {}".format(a, b, c) for a, b, c in
                        zip(instruction, history, cand_text)]

            text_input = self.lang_model.tokenize(all_text).to(fuse_embeds.device)

            cand_embeds = fuse_embeds[cand_masks]  # .to(self.model_type)
            loss, logits, hidden_states = self.lang_model(
                input_ids=text_input['input_ids'],
                attention_mask=text_input['attention_mask'],
                cand_vis=cand_embeds,
                hist_vis=hist_vis_input,
            )

            fuse_logits = torch.zeros((fuse_embeds.shape[0], fuse_embeds.shape[1])).to(
                fuse_embeds.device).to(self.model_type)

            fuse_logits[cand_masks] = self.out_head(hidden_states).squeeze()

            fuse_logits.masked_fill_(cand_masks.logical_not(), -float('inf'))
            fuse_logits.masked_fill_(gmap_visited_masks, -float('inf'))

            return {
                'fuse_embeds': fuse_embeds.detach(),
                'fuse_logits': fuse_logits,
            }


def get_vlnbert_models_ddp(args, config=None):
    from transformers import PretrainedConfig

    from transformers.utils import logging
    logging.set_verbosity_error()

    model_name_or_path = None  # args.bert_ckpt_file
    new_ckpt_weights = {}

    cfg_name = 'bert-base-uncased'

    vis_config = PretrainedConfig.from_pretrained(cfg_name)

    vis_config.precision = args.precision
    vis_config.tokenizer_path = args.tokenizer_path
    vis_config.max_action_steps = 100
    vis_config.image_feat_size = args.image_feat_size
    vis_config.angle_feat_size = args.angle_feat_size
    vis_config.obj_feat_size = args.obj_feat_size
    vis_config.obj_loc_size = 3
    vis_config.num_l_layers = args.num_l_layers
    vis_config.num_pano_layers = args.num_pano_layers
    vis_config.num_x_layers = args.num_x_layers
    vis_config.graph_sprels = args.graph_sprels
    vis_config.glocal_fuse = args.fusion == 'dynamic'

    vis_config.fix_lang_embedding = False  # args.fix_lang_embedding
    vis_config.fix_pano_embedding = False  # args.fix_pano_embedding
    vis_config.fix_local_branch = False  # args.fix_local_branch

    vis_config.update_lang_bert = True  # not args.fix_lang_embedding
    vis_config.output_attentions = True
    vis_config.pred_head_dropout_prob = 0.1
    vis_config.use_lang2visn_attn = False
    new_ckpt_weights = {}
    visual_model = GlocalTextPathNavCMT.from_pretrained(
        pretrained_model_name_or_path=None,
        config=vis_config,
        state_dict=new_ckpt_weights
    )

    return visual_model


class VLNBert(nn.Module):
    def __init__(self, args, use_ddp=False):
        super().__init__()
        print('\nInitalizing the VLN-BERT model ...')
        self.args = args

        if use_ddp:
            self.vln_bert = get_vlnbert_models_ddp(args, config=None)
        else:
            raise NotImplementedError
        self.drop_env = nn.Dropout(p=args.feat_dropout)

    def forward(self, mode, batch):
        batch = collections.defaultdict(lambda: None, batch)

        if mode == 'language':
            self.vln_bert(mode, batch)
            return

        elif mode == 'panorama':  # batch['view_img_fts'] [B, 36, D=768] --> dropout
            batch['view_img_fts'] = self.drop_env(batch['view_img_fts'])
            if 'obj_img_fts' in batch:  # False
                batch['obj_img_fts'] = self.drop_env(batch['obj_img_fts'])
            pano_embeds, pano_masks = self.vln_bert(mode, batch)
            return pano_embeds, pano_masks

        elif mode == 'navigation':
            outs = self.vln_bert(mode, batch)
            return outs

        else:
            raise NotImplementedError('wrong mode: %s' % mode)


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()
