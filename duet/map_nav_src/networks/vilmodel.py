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

from .ops import create_transformer_encoder
from .ops import extend_neg_masks, gen_seq_masks, pad_tensors_wgrad
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


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

class LangModel(nn.Module):
    def __init__(self):
        super().__init__()
        path = '/mnt/lustre/huangshijia.p/LLAMA_7B'
        path = '/mnt/petrelfs/chenzhi/workspace/LLM/models/Vicuna-7B'
        self.tokenizer = LlamaTokenizer.from_pretrained(path)
        # path = 'bigscience/bloom-560m'
        # self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.tokenizer.padding_side = 'left'

        self.cand_token = ['<cand>']
        self.hist_token = ['<hist>']

        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.cand_token + self.hist_token}
        )

        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.cand_token_id = self.tokenizer.encode("".join(self.cand_token), add_special_tokens=False)
        self.hist_token_id = self.tokenizer.encode("".join(self.hist_token), add_special_tokens=False)

        self.lang_model = AutoModelForCausalLM.from_pretrained(path).bfloat16()
        self.lang_model.resize_token_embeddings(len(self.tokenizer))
        dim = 4096
        # dim = 1024
        self.mapper = nn.Linear(768, dim)

    def tokenize(self, text):
        batch_text = self.tokenizer(
            text,
            max_length=1024,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
            return_token_type_ids=True
        )
        return batch_text

    def forward(self, *input, **kwargs):
        input_ids = kwargs["input_ids"] if "input_ids" in kwargs else input[0] # B,L

        hist_locations = (input_ids >= self.hist_token_id[0]) & (input_ids <= self.hist_token_id[-1])
        cand_locations = (input_ids >= self.cand_token_id[0]) & (input_ids <= self.cand_token_id[-1])

        inputs_embeds = self.lang_model.model.embed_tokens(input_ids)
        # inputs_embeds = self.lang_model.transformer.word_embeddings(input_ids)

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
        # outputs = self.lang_model.transformer(*input, **kwargs)

        hidden_states = outputs[0]
        logits = self.lang_model.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
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
                traj_obj_img_embeds = self.obj_layer_norm(self.obj_linear(traj_obj_img_embeds))
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

class GlocalTextPathNavCMT(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = 4096
        # dim = 1024
        self.img_embeddings = ImageEmbeddings(config)
        self.lang_model = LangModel()
        self.vp_pos_embeddings = nn.Sequential(
            nn.Linear(4+4+3, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.local_sap_head = ClsPrediction(dim).bfloat16()


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

    def random_permute(self, cand_masks, do):
        cand_num = cand_masks.sum(dim=-1).tolist()
        fws = []
        bws = []
        acc_num = 0
        for num in cand_num:
            if do:
                fw = np.array([0] + (np.random.permutation(num-1)+1).tolist())
            else:
                fw = np.array([i for i in range(num)])
            bw = [i for i in range(num)]
            bw = np.array(bw)
            bw[fw] = [i for i in range(num)]
            fw = fw+acc_num
            bw = bw+acc_num
            acc_num += num
            fws.extend(fw.tolist())
            bws.extend(bw.tolist())
        return fws, bws

    def forward(self, mode, batch, **kwargs):
        if mode == 'language':
            return

        elif mode == 'panorama':
            pano_embeds, pano_masks = self.forward_panorama_per_step(
                batch['view_img_fts'], batch['obj_img_fts'], batch['loc_fts'],
                batch['nav_types'], batch['view_lens'], batch['obj_lens']
            )
            return pano_embeds, pano_masks

        elif mode == 'qa':
            vp_img_embeds = batch['vp_img_embeds']
            vp_pos_fts = batch['vp_pos_fts']
            cand_masks = batch['vp_nav_masks']
            
            vp_embeds = vp_img_embeds + self.vp_pos_embeddings(vp_pos_fts)
            cand_nums = cand_masks.sum(dim=-1)

            nav_text = []
            for nav_num in cand_nums:
                nav_text.append(''.join([' ({}) <cand>'.format(i) for i in range(nav_num)]))
            all_text = []
            for a, b, c in zip(nav_text, batch['qas'], batch['gt_id']):
                cnt_text = ""
                # cnt_text += '### Instruction: {} \n'.format("")
                # cnt_text += '### History: {} \n'.format("")
                cnt_text += '### Candidate: {} \n'.format(a)
                cnt_text += 'what is the navigation action corresponding to candidate ({}) ?\n'.format(c)
                cnt_text += 'Output: {}'.format("")
                all_text.append([cnt_text,b])

            text_input = self.lang_model.tokenize(all_text).to(vp_embeds.device)
            labels = text_input['input_ids'].clone()
            labels[text_input['token_type_ids'][:,-labels.shape[-1]:]==0] = -100
            #[332,4566]
            loss, logits, hidden_states = self.lang_model(
                input_ids = text_input['input_ids'],
                attention_mask = text_input['attention_mask'],
                labels = labels,
                cand_vis = vp_embeds[cand_masks],
            )
            # import pdb;pdb.set_trace()
            return loss

        elif mode == 'sum':
            
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

            hist_nums = [len(his) for his in history]
            hist_text = []
            for hist_num in hist_nums:
                hist_text.append(''.join([' ({}) <hist>'.format(i) for i in range(hist_num)]))

            all_text = []
            for a,b in zip(instruction, hist_text):
                cnt_text = ""
                cnt_text += '### History: {} \n'.format(b)
                cnt_text += 'what is the navigation action corresponding to the navigation history?\n'
                cnt_text += 'Output: {}'.format("")
                all_text.append([cnt_text,a])

            text_input = self.lang_model.tokenize(all_text).to(hist_vis_input.device)
            labels = text_input['input_ids'].clone()
            labels[text_input['token_type_ids'][:,-labels.shape[-1]:]==0] = -100

            loss, logits, hidden_states = self.lang_model(
                input_ids=text_input['input_ids'],
                attention_mask=text_input['attention_mask'],
                labels=labels,
                hist_vis=hist_vis_input,
            )
            return loss

        elif mode == 'navigation':
            vp_img_embeds = batch['vp_img_embeds']
            vp_pos_fts = batch['vp_pos_fts']
            nav_masks = batch['vp_nav_masks']
            if 'vp_obj_masks' in batch:
                obj_masks = batch['vp_obj_masks']
            else:
                obj_masks = torch.zeros_like(nav_masks)
            cand_masks = nav_masks | obj_masks

            vp_embeds = vp_img_embeds + self.vp_pos_embeddings(vp_pos_fts)

            nav_nums = nav_masks.sum(dim=-1)
            obj_nums = obj_masks.sum(dim=-1)
            cand_nums = cand_masks.sum(dim=-1)

            idx_mapper = self.random_permute(cand_masks, do=batch['do_permute'])
            # import pdb;pdb.set_trace()

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

            hist_nums = [len(his) for his in history]
            nav_text = []
            obj_text = []
            hist_text = []
            for nav_num, obj_num, hist_num in zip(nav_nums, obj_nums, hist_nums):
                nav_text.append(''.join([' ({}) <cand>'.format(i) for i in range(nav_num)]))
                obj_text.append(''.join([' ({}) <cand>'.format(i) for i in range(obj_num)]))
                hist_text.append(''.join([' ({}) <hist>'.format(i) for i in range(hist_num)]))

            all_text = []
            for a,b,c in zip(instruction, hist_text, nav_text):
                cnt_text = ""
                # cnt_text += 'Given the instruction, you should decide which direction to go at each step and finally arrive at the target location.\n'
                cnt_text += 'Following is the History, which contains the visual information of your previous decisions.\n'
                cnt_text += '### History: {} \n'.format(b)
                cnt_text += 'Following is the Candidate, which contains several directions you can go to at the current position, candidate (0) is stop.\n'
                cnt_text += '### Candidate: {} \n'.format(c)
                cnt_text += '### Instruction: {} \n'.format(a)
                cnt_text += 'Based on the instruction and history, select the correct direction from the candidates to go to the target location.\n'
                cnt_text += 'Output: candidate (1'
                all_text.append(cnt_text)

            text_input = self.lang_model.tokenize(all_text).to(vp_embeds.device)

            loss, logits, hidden_states = self.lang_model(
                input_ids = text_input['input_ids'],
                attention_mask = text_input['attention_mask'],
                cand_vis = vp_embeds[cand_masks][idx_mapper[0]],
                hist_vis = hist_vis_input,
            )
            # logits = logits.softmax(dim=-1)
            vp_score1 = self.local_sap_head(hidden_states).squeeze()
            
            dd = [29900, 29896, 29906, 29941, 29946, 29945, 29953, 29955, 29947, 29929]
            vp_score_s1 = logits[:, -2, dd]
            vp_score_s1[:, 1] = (vp_score_s1[:, 1] + logits[:, -1, 29897]) / 2
            vp_score_s2 = (logits[:, -1, dd] + logits[:, -2, dd[1]][...,None]) / 2
            vp_score = torch.cat([vp_score_s1, vp_score_s2],dim=-1)

            lm_score = [score[:num] for score, num in zip(vp_score, cand_masks.sum(dim=-1).tolist())]
            lm_score = torch.cat(lm_score, dim=0)

            # import pdb;pdb.set_trace()
            local_logits = torch.zeros((vp_embeds.shape[0], vp_embeds.shape[1])).to(vp_embeds.device).bfloat16()
            local_logits[cand_masks] += lm_score[idx_mapper[1]]
            # local_logits[cand_masks] += vp_score1[idx_mapper[1]]
            local_logits.masked_fill_(nav_masks.logical_not(), -float('inf'))
            
            obj_logits = torch.zeros((vp_embeds.shape[0], vp_embeds.shape[1])).to(vp_embeds.device).bfloat16()
            # obj_logits[..., :20] = vp_score[..., :20]
            # obj_logits[cand_masks] += vp_score1
            # obj_logits.masked_fill_(obj_masks.logical_not(), -float('inf'))

            # vp_embeds = vp_embeds[nav_masks].
            vp_embeds = ((vp_embeds * nav_masks[..., None]).sum(dim=1) / nav_masks.sum(dim=-1)[..., None]).detach() 
            return {
                'vp_embeds': vp_embeds,
                'local_logits': local_logits,
                'obj_logits': obj_logits,
            }

# 