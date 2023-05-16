import random
from typing import List
import torch.nn as nn

from .helpers import GatedCrossAttentionBlock
from .utils import getattr_recursive, setattr_recursive


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

        self.media_token_id = media_token_id
        self.initialized_flamingo = True
        self.state_token_id = state_token_id

    def forward(self, *input, **kwargs):
        """Condition the Flamingo layers on the media locations before forward()"""
        if not self.initialized_flamingo:
            raise ValueError(
                "Flamingo layers are not initialized. Please call `init_flamingo` first."
            )

        if kwargs["past_key_values"] is None:
            input_ids = kwargs["input_ids"] if "input_ids" in kwargs else input[0]

            if isinstance(self.media_token_id, int):
                media_locations = input_ids == self.media_token_id
            elif isinstance(self.media_token_id, list):
                media_locations = (input_ids >= self.media_token_id[0]) & \
                                  (input_ids <= self.media_token_id[-1])

            inputs_embeds = self.model.embed_tokens(input_ids)
            inputs_embeds[media_locations] += self.vis_x

            kwargs["input_ids"] = None
            kwargs["inputs_embeds"] = inputs_embeds

        return super().forward(
            *input, **kwargs
        )  # Call the other parent's forward method

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
