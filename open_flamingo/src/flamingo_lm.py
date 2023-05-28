import random
from typing import List
import torch.nn as nn

from .helpers import GatedCrossAttentionBlock
from .utils import getattr_recursive, setattr_recursive
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast

class FlamingoLMMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def condition_vis_x(self, vis_x):
        self.vis_x = vis_x

    def set_history_state(self, state):
        self.state = state

    def init_flamingo(
        self,
        media_token_id,
        vis_hidden_size,
        cross_attn_every_n_layers,
        use_media_placement_augmentation,
        state_token_id,
    ):
        """
        Initialize Flamingo by adding a new gated cross attn to the decoder.
        Store the media token id for computing the media locations.
        """
        self.action_head = nn.Linear(4096, 1)
        self.media_token_id = media_token_id
        self.initialized_flamingo = True
        self.state_token_id = state_token_id

    def forward(self, *input, **kwargs):
        """Condition the Flamingo layers on the media locations before forward()"""
        if not self.initialized_flamingo:
            raise ValueError(
                "Flamingo layers are not initialized. Please call `init_flamingo` first."
            )

        # if kwargs["past_key_values"] is None:
        input_ids = kwargs["input_ids"] if "input_ids" in kwargs else input[0] # B,L
        seq_len = input_ids.shape[1]
        bs = input_ids.shape[0]

        media_locations = (input_ids == self.media_token_id[0])

        # <img><action-bos><lang-bos></s>
        token_shortcuts = []
        for media_location, input_id in zip(media_locations, input_ids):
            special_locations = media_location | (input_id == self.media_token_id[1])
            special_locations = special_locations.nonzero().squeeze()
            token_shortcut = {}
            beg = 0
            end = 0
            for location in special_locations:
                if input_id[location] == self.media_token_id[1]:
                    token_shortcut[location.item()-seq_len] = (beg, end)
                    beg = end
                else:
                    end += 1
            token_shortcuts.append(token_shortcut)

        inputs_embeds = self.model.embed_tokens(input_ids)
        inputs_embeds[media_locations] += self.vis_x
        kwargs["input_ids"] = None
        kwargs["inputs_embeds"] = inputs_embeds
    
        labels = None
        if 'labels' in kwargs:
            labels = kwargs.pop('labels')
        outputs = self.model(*input, **kwargs)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        for idx in range(bs):
            token_shortcut = token_shortcuts[idx]
            action_logit = self.action_head(hidden_states[idx][media_locations[idx]]).squeeze()
            for shortcut in token_shortcut:
                sa, sb = token_shortcut[shortcut]
                sn = sb - sa
                logits[idx][shortcut][self.media_token_id[2]:self.media_token_id[2]+sn] += action_logit[sa:sb]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def is_conditioned(self) -> bool:
        return True
        """Check whether all decoder layers are already conditioned."""
        # return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clear_conditioned_layers(self):
        self.condition_vis_x(None)
        # for layer in self._get_decoder_layers():
        #     layer.condition_vis_x(None)
        #     layer.condition_media_locations(None)
        #     layer.condition_attend_previous(None)
