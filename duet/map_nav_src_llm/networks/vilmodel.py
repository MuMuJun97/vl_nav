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

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    # logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
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
        config.intermediate_size = 768
        config.num_hidden_layers = 2
        config.hidden_size = 768
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
    def __init__(self):
        super().__init__()

        if 'lustre' in __file__:
            path = '/mnt/lustre/share_data/huangshijia/alpaca'
            self.tokenizer = AutoTokenizer.from_pretrained(
                path, local_files_only=False
            )
            is_s2_server = True
        else:
            tokenizer_path = '/home/zlin/vln/llm/alpaca_model/model_config.pkl'
            path = Path(tokenizer_path).parent.resolve().__str__()
            self.tokenizer = AutoTokenizer.from_pretrained(
                path, local_files_only=False
            )
            is_s2_server = False

        # path = '/mnt/lustre/huangshijia.p/LLAMA_7B'
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     path, local_files_only=True
        # )
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

        if is_s2_server:
            self.lang_model = AutoModelForCausalLM.from_pretrained(
                path, local_files_only=False
            ).bfloat16()
        else:
            self.lang_model = llama_model_in_debug_model(tokenizer_path)
            self.lang_model = self.lang_model.bfloat16()

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

        inputs_embeds = self.lang_model.model.embed_tokens(input_ids)
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
        outputs = self.lang_model.model(*input, **kwargs)
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

        # logits = logits[cand_locations]
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


class GlocalTextPathNavCMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.img_embeddings = ImageEmbeddings(config)
        self.lang_model = LangModel()
        self.vp_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size*2, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )

        self.local_sap_head = ClsPrediction(self.lang_model.hidden_size).bfloat16()
        if self.config.obj_feat_size > 0:
            self.og_head = ClsPrediction(self.lang_model.hidden_size).bfloat16()

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
                      self.img_embeddings.nav_type_embedding(nav_types)
        pano_embeds = self.img_embeddings.layer_norm(pano_embeds)
        pano_embeds = self.img_embeddings.dropout(pano_embeds)

        pano_masks = gen_seq_masks(pano_lens)
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
            return pano_embeds, pano_masks

        elif mode == 'navigation':
            vp_img_embeds = batch['vp_img_embeds']
            vp_pos_fts = batch['vp_pos_fts']
            vp_embeds = vp_img_embeds + self.vp_pos_embeddings(vp_pos_fts)

            cand_masks = batch['vp_masks'] & batch['vp_nav_masks']
            cand_nums = cand_masks.sum(dim=-1)

            instruction = batch['instruction']
            history = batch['history']
            hist_vis = batch['hist_vis']
            hist_vis_input = []
            for vis in hist_vis:
                hist_vis_input.extend(vis)
            if hist_vis_input != []:
                hist_vis_input = torch.stack(hist_vis_input,dim=0)
            else:
                hist_vis_input = None

            cand_text = []
            for idx, cand_num in enumerate(cand_nums):
                cand_text.append(''.join(['<cand>'.format(i) for i in range(cand_num)]))
            cand_text = ['\nEnvironment: ' + text + '\nAgent: <s>' for text in cand_text]

            # for instr, old_info, c_txt in zip(instruction['instruction'], history)

            all_text = ["Conmander: {} \nHistory: {} {}".format(a,b,c) for a,b,c in zip(instruction, history, cand_text)]

            text_input = self.lang_model.tokenize(all_text).to(vp_embeds.device)
            loss, logits, hidden_states = self.lang_model(
                input_ids = text_input['input_ids'],
                attention_mask = text_input['attention_mask'],
                cand_vis = vp_embeds[cand_masks],
                hist_vis = hist_vis_input,
            )
            local_logits = torch.zeros((vp_embeds.shape[0], vp_embeds.shape[1])).to(vp_embeds.device).bfloat16()
            local_logits.masked_fill_(cand_masks.logical_not(), -float('inf'))
            local_logits[cand_masks] = self.local_sap_head(hidden_states).squeeze()
            return {
                'vp_embeds': vp_embeds.detach(),
                'local_logits': local_logits,
            }

            
       