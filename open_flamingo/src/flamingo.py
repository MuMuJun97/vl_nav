import torch
from einops import rearrange
from torch import nn
from transformers.utils import ModelOutput
from typing import Any, Dict, List, Optional, Tuple, Union
from .helpers import PerceiverResampler


class Flamingo(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_encoder: nn.Module,
        eoc_token_id: int,
        media_token_id,
        vis_dim: int,
        cross_attn_every_n_layers: int = 1,
        use_media_placement_augmentation: bool = False,
        view_nums: int = 12,
        history_vision: bool = False,
        state_token_id: int = 1,
        multi_state: bool = False,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            eoc_token_id (int): Token id for <|endofchunk|>
            media_token_id (int): Token id for <image>
            vis_dim (int): Dimension of the visual features.
                Visual features are projected to match this shape along the last dimension.
            cross_attn_every_n_layers (int, optional): How often to apply cross attention after transformer layer. Defaults to 1.
            use_media_placement_augmentation (bool, optional): Whether to randomly assign images to the preceding or following text in training. Defaults to False.
        """
        super().__init__()
        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id
        self.use_media_placement_augmentation = use_media_placement_augmentation
        self.vis_dim = vis_dim
        self.vision_encoder = vision_encoder
        # self.perceiver = PerceiverResampler(dim=self.vis_dim)

        self.lang_encoder = lang_encoder

        self.lang_encoder.init_flamingo(
            media_token_id=media_token_id,
            vis_hidden_size=self.vis_dim,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            use_media_placement_augmentation=self.use_media_placement_augmentation,
            state_token_id=state_token_id,
        )

        # @1 self.vis_dim: CLIP image feature dim;
        # @2 LLaMa, hidden size
        self.mapper = nn.Sequential(
            nn.Linear(self.vis_dim, self.lang_encoder.config.hidden_size),
            nn.LayerNorm(self.lang_encoder.config.hidden_size)
        )

        self.angle_encoder = nn.Sequential(
            nn.Linear(2, self.lang_encoder.config.hidden_size),
            nn.LayerNorm(self.lang_encoder.config.hidden_size)
        )

    def forward_train(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        use_cached_vision_x: bool = False,
        clear_conditioned_layers: bool = True,
        past_key_values=None,
        use_cache: bool = False,
        use_local_vision: str = 'none',
        history_vis: int = -1,
    ):
        """
        Forward pass of Flamingo.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
                Batch_size, T_img: num_media=12, F: num_frames
            lang_x (torch.Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
            use_cache: whether to use cached key values. See use_cache
                documentation in Hugging Face CausalLM models.
            use_local_vision: VLN-DUET local image features.
            history_vis: vision images - history state
        """

        if use_local_vision == 'feature':
            self._encode_vision_with_local(vision_x)
        elif use_local_vision == 'image':
            self._encode_multi_view_image(vision_x)
        else:
            # multi-step dialog: vision + mask
            if isinstance(vision_x, tuple):
                self.encode_image_with_mask(vision_x)
            else:
                self._encode_vision_x(vision_x=vision_x)
   
        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()

        return output

    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        use_cached_vision_x: bool = False,
        clear_conditioned_layers: bool = True,
        past_key_values=None,
        use_cache: bool = False,
        use_local_vision: str = 'none',
        mode: str = 'train',
        max_length: int = 20,
        history_vis: int = -1,
    ):
        if mode == 'train':
            return self.forward_train(
                vision_x=vision_x,
                lang_x=lang_x,
                attention_mask=attention_mask,
                labels=labels,
                use_cached_vision_x=use_cached_vision_x,
                clear_conditioned_layers=clear_conditioned_layers,
                past_key_values=past_key_values,
                use_cache=use_cache,
                use_local_vision=use_local_vision,
                history_vis=history_vis,
            )
        elif mode == 'generate':
            output_ids = self.greedy_inference(
                vision_x=vision_x,
                input_ids=lang_x,
                attention_mask=attention_mask,
                labels=labels,
                use_cached_vision_x=use_cached_vision_x,
                clear_conditioned_layers=clear_conditioned_layers,
                use_local_vision=use_local_vision,
                max_length=max_length,
            )
            return output_ids

    def _encode_vision_with_local(self, vision_x: torch.Tensor):
        """
        :param vision_x: torch.Size([B, 12, 768])
        :return:
        """
        vision_x = vision_x.unsqueeze(1).unsqueeze(1)
        raise NotImplementedError

    def _encode_multi_view_image(self, all_vision_x: torch.Tensor):
        """
        Args:
            vision_x (torch.Tensor): Vision input
        """
        assert all_vision_x.ndim == 7, "vision_x should be of shape (b, M, T_img, F, C, H, W)"
        b, M, T, F = all_vision_x.shape[:4] # Batch size, Multi Views, 1, 1
        assert F == 1, "Only single frame supported"

        all_vision_x = rearrange(all_vision_x, "b M T F c h w -> (b M) T F c h w")

        vision_x = all_vision_x
        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            vision_x = self.vision_encoder.visual(vision_x)[1]  # (B*M,256,1024)
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b*M, T=T, F=F)
        vision_x = self.perceiver(vision_x)  # reshapes to (b*M, T, n, d) (b*M,1,64,1024)

        # TODO: Multi-View Fusion
        vision_x = rearrange(vision_x, "(b M) t v d -> b M t v d", b=b, M=M) # (b, 12, 1, 64, 1024)
        avg_pano_vision_x = torch.mean(vision_x,dim=1)

        self.lang_encoder.condition_vis_x(avg_pano_vision_x)
        # for layer in self.lang_encoder._get_decoder_layers():
        #     layer.condition_vis_x(avg_pano_vision_x)

        # all_vision_feats = []
        # for m in range(M):
        #     vision_x = all_vision_x[:,m,...].contiguous()
        #     vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        #     with torch.no_grad():
        #         vision_x = self.vision_encoder.visual(vision_x)[1] # (B,256,1024)
        #     vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        #     vision_x = self.perceiver(vision_x)  # reshapes to (b, T, n, d)
        #
        #     # Multi-View: 12 * (B,1,64,1024)
        #     all_vision_feats.append(vision_x)

    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            vision_x = self.vision_encoder.visual(vision_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)

        vision_x = self.perceiver(vision_x)  # reshapes to (b, T, n, d)
        vision_x = self.mapper(vision_x.mean(dim=-2))

        # view embedding
        view_id_tensor = torch.arange(vision_x.shape[1], device=vision_x.device).repeat((vision_x.shape[0], 1))
        view_id_embeds = self.img_id_embedding(view_id_tensor)
        view_vision_x = vision_x + view_id_embeds

        self.lang_encoder.condition_vis_x(view_vision_x)
        # for layer in self.lang_encoder._get_decoder_layers():
        #     layer.condition_vis_x(vision_x)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):

        model_inputs = {"input_ids": input_ids}


        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = outputs.past_key_values

        # # update token_type_ids with last value
        # if "token_type_ids" in model_kwargs:
        #     token_type_ids = model_kwargs["token_type_ids"]
        #     model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # if "attention_mask" in model_kwargs:
        #     attention_mask = model_kwargs["attention_mask"]
        #     model_kwargs["attention_mask"] = torch.cat(
        #         [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
        #     )


        return model_kwargs

    def greedy_inference(
            self,
            vision_x: torch.Tensor,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None,
            labels: torch.Tensor = None,
            use_cached_vision_x: bool = False,
            clear_conditioned_layers: bool = True,
            past_key_values=None,
            use_cache: bool = False,
            use_local_vision: str = 'none',
            max_length: int = 20,
            model_kwargs: dict = {},
    ):
        
        model_kwargs['use_cache'] = True

        output_attentions = False
        output_hidden_states = False
        scores = None
        max_length += input_ids.shape[-1]

        self.encode_image_with_mask(vision_x=vision_x)

        pad_token_id = self.lang_encoder.generation_config.pad_token_id
        eos_token_id = self.eoc_token_id # STOP Token
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        new_input_ids = input_ids

        while True:
            # prepare model inputs
            if 'attention_mask' not in model_kwargs:
                model_kwargs['attention_mask'] = attention_mask
            else:
                model_kwargs['attention_mask'] = torch.cat([model_kwargs['attention_mask'], attention_mask],dim=1)

            model_inputs = self.prepare_inputs_for_generation(new_input_ids, **model_kwargs)
            # forward pass to get next token
            outputs = self.lang_encoder(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            next_tokens_scores = outputs.logits[:, -1, :]
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            new_input_ids = next_tokens[:, None]
            attention_mask = attention_mask.new_ones((attention_mask.shape[0], 1))

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False
            )

            # stop when each sentence is finished, or if we exceed the maximum length
            if next_tokens[0] in eos_token_id or input_ids.shape[-1] >= max_length:
                break

        self.lang_encoder.clear_conditioned_layers()

        return input_ids, model_kwargs

    def encode_image_with_mask(self, vision_x):
        """
        Args:
            vision_x: tuple (vision_x, image_mask)

        Returns:

        """
        with torch.no_grad():
            input_angle_feats = vision_x[2]
            image_mask = vision_x[1].bool()
            vision_x = vision_x[0]

            assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
            b, T, F = vision_x.shape[:3]
            assert F == 1, "Only single frame supported"

            # vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
            vision_x = vision_x[image_mask].flatten(0,1)
            vision_x = self.vision_encoder.visual(vision_x)[1]
            vision_x = vision_x.mean(dim=-2)
            input_angle_feats = input_angle_feats[image_mask]
            # vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)

        vision_x = self.mapper(vision_x)
        angle_feats = self.angle_encoder(input_angle_feats)

        self.lang_encoder.condition_vis_x(vision_x + angle_feats)
