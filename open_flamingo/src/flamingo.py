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
        media_token_id: int,
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
        self.perceiver = PerceiverResampler(dim=self.vis_dim)

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
        # @param: img views
        self.view_nums = view_nums
        self.img_id_embedding = nn.Embedding(view_nums, self.lang_encoder.config.hidden_size)

        self.have_state = False
        self.history_vision = history_vision
        self.multi_state = multi_state
        if history_vision:
            if multi_state:
                self.history_encoder = nn.Sequential(
                    nn.Linear(self.lang_encoder.config.hidden_size, self.lang_encoder.config.hidden_size),
                    nn.LayerNorm(self.lang_encoder.config.hidden_size)
                )
            else:
                self.history_encoder = nn.Sequential(
                    nn.Linear(self.lang_encoder.config.hidden_size*2, self.lang_encoder.config.hidden_size),
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
        if history_vis != -1:
            self.encode_vision_x_with_state(
                vision_x=vision_x,
                history_vis=history_vis,
            )
        else:
            if use_local_vision == 'feature':
                self._encode_vision_with_local(vision_x)
            elif use_local_vision == 'image':
                self._encode_multi_view_image(vision_x)
            else:
                assert (
                               vision_x is not None
                       ) or use_cached_vision_x, (
                    "Must provide either vision_x or use_cached_vision_x to True."
                )

                if use_cached_vision_x:
                    # Case: use cached; vision_x should be cached and other
                    # vision-related inputs should not be provided.
                    assert (
                            vision_x is None
                    ), "Expect vision_x to be None when use_cached_vision_x is True."
                    assert self.lang_encoder.is_conditioned()

                else:
                    # Case: do not use caching (i.e. this is a standard forward pass);
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
            # self.lang_encoder.clear_history_state()

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
        history_vis: bool = False,
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
                past_key_values=past_key_values,
                use_cache=use_cache,
                use_local_vision=use_local_vision,
                max_length=max_length,
            )
            return output_ids

    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        num_beams=1,
        max_new_tokens=None,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        no_repeat_ngram_size=0,
        prefix_allowed_tokens_fn=None,
        length_penalty=1.0,
        num_return_sequences=1,
        do_sample=False,
        early_stopping=False,
    ):
        """
        Generate text conditioned on vision and language inputs.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                images in the same chunk are collated along T_img, and frames are collated along F
                currently only F=1 is supported (single-frame videos)
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            max_length (int, optional): Maximum length of the output. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            num_beams (int, optional): Number of beams. Defaults to 1.
            max_new_tokens (int, optional): Maximum new tokens. Defaults to None.
            temperature (float, optional): Temperature. Defaults to 1.0.
            top_k (int, optional): Top k. Defaults to 0.
            top_p (float, optional): Top p. Defaults to 1.0.
            no_repeat_ngram_size (int, optional): No repeat ngram size. Defaults to 0.
            length_penalty (float, optional): Length penalty. Defaults to 1.0.
            num_return_sequences (int, optional): Number of return sequences. Defaults to 1.
            do_sample (bool, optional): Do sample. Defaults to False.
            early_stopping (bool, optional): Early stopping. Defaults to False.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        if num_beams > 1:
            vision_x = vision_x.repeat_interleave(num_beams, dim=0)

        self._encode_vision_x(vision_x=vision_x)

        output = self.lang_encoder.generate(
            lang_x,
            attention_mask=attention_mask,
            eos_token_id=self.eoc_token_id,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            early_stopping=early_stopping,
        )

        self.lang_encoder.clear_conditioned_layers()
        return output

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

    def encode_vision_x_with_state(self, vision_x: torch.Tensor, history_vis: int):
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

        self.lang_encoder.condition_vis_x(view_vision_x,has_state=True)

        if history_vis == 0:
            self.have_state = False
            self.lang_encoder.clear_history_state()

        if not self.have_state:
            self.have_state = True
            if self.multi_state:
                vision_state = self.history_encoder(
                    view_vision_x.mean(dim=-2).unsqueeze(1)
                )
                self.lang_encoder.set_history_state(vision_state)
            else:
                self.lang_encoder.set_history_state(view_vision_x.mean(dim=-2))
        else:
            history_vision = self.lang_encoder.get_history_state()
            if self.multi_state:
                vision_state = self.history_encoder(
                    view_vision_x.mean(dim=-2).unsqueeze(1)
                )
                vision_state = torch.cat([vision_state,history_vision],dim=1)
            else:
                vision_state = torch.cat([view_vision_x.mean(dim=-2), history_vision], dim=-1)
                vision_state = self.history_encoder(vision_state)
            self.lang_encoder.set_history_state(vision_state)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
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

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            raise NotImplementedError

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
    ):
        model_kwargs = {
            'attention_mask': attention_mask,
            'use_cache': True,
        }
        output_attentions = False
        output_hidden_states = False
        scores = None
        max_length += input_ids.shape[-1]

        self._encode_vision_x(vision_x=vision_x)

        pad_token_id = self.lang_encoder.generation_config.pad_token_id
        eos_token_id = self.eoc_token_id # STOP Token
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        while True:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            outputs = self.lang_encoder(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            next_token_logits = outputs.logits[:, -1, :]
            next_tokens_scores = next_token_logits

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                # torch.ne: not equal
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or input_ids.shape[-1] >= max_length:
                break

        self.lang_encoder.clear_conditioned_layers()

        return input_ids