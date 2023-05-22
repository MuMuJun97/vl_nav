from transformers import AutoModelForCausalLM, AutoTokenizer
import open_clip

from .flamingo import Flamingo
from .flamingo_lm import FlamingoLMMixin
from .utils import extend_instance

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
        config.intermediate_size = 512
        config.num_hidden_layers = 2
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


def create_model_and_transforms(
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    lang_encoder_path: str,
    tokenizer_path: str,
    enable_offline_vision_encoder: bool = False,
    cross_attn_every_n_layers: int = 1,
    use_local_files: bool = False,
    decoder_layers_attr_name: str = None,
    args=None,
    **flamingo_kwargs,
):
    """
    Initialize a Flamingo model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        clip_vision_encoder_path (str): path to pretrained clip model (e.g. "ViT-B-32")
        clip_vision_encoder_pretrained (str): name of pretraining dataset for clip model (e.g. "laion2b_s32b_b79k")
        lang_encoder_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        enable_offline_vision_encoder (bool, optional): 使用local image view features.
        cross_attn_every_n_layers (int, optional): determines how often to add a cross-attention layer. Defaults to 1.
        use_local_files (bool, optional): whether to use local files. Defaults to False.
        decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
        args:
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """
    r2r_tok = args.r2r_tok # (bool, optional): 是否添加新的tokens.
    if 'model_config.pkl' in tokenizer_path:
        from pathlib import Path
        tokenizer_path = Path(tokenizer_path).parent.resolve().__str__()
    if enable_offline_vision_encoder:
        vision_encoder = None
        image_processor = None
    else:
        vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
            clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained
        )
        # set the vision encoder to output the visual features
        vision_encoder.visual.output_tokens = True

    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=use_local_files
    )
    # add Flamingo special tokens to the tokenizer

    if r2r_tok:
        # add <walkto0-11>
        action_tokens = ['<image{}>'.format(x) for x in range(12)] \
                        + ['<walkto{}>'.format(_) for _ in range(12)] + ['<stop>']
        media_token_id = None
    else:
        action_tokens = ["<|endofchunk|>", "<image>", "<state>"]
        media_token_id = text_tokenizer.encode("<image>")[-1]

    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": action_tokens}
    )
    if text_tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    # TODO: only for debug
    if 'model_config.pkl' in lang_encoder_path:
        lang_encoder = llama_model_in_debug_model(lang_encoder_path)
    else:
        lang_encoder = AutoModelForCausalLM.from_pretrained(
            lang_encoder_path, local_files_only=use_local_files
        )

    extend_instance(lang_encoder, FlamingoLMMixin)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))

    if flamingo_kwargs.get('unfreeze_llm',None) is not None:
        unfreeze_llm = flamingo_kwargs['unfreeze_llm']
        flamingo_kwargs.pop('unfreeze_llm')
    else:
        unfreeze_llm = False

    lang_encoder = lang_encoder.bfloat16()
    # TODO ? endofchunk: how to modify?
    # cross_attn_every_n_layers: multi-modal cross fusion layer.

    if r2r_tok:
        image_tokens = ['<image{}>'.format(x) for x in range(12)]
        media_token_id = text_tokenizer.encode(
            "".join(image_tokens), add_special_tokens=False
        )

    model = Flamingo(
        vision_encoder,
        lang_encoder,
        text_tokenizer.encode("</s>"+"".join(action_tokens), add_special_tokens=False),
        media_token_id=media_token_id,
        vis_dim=open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"][
            "width"
        ],
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        history_vision=r2r_tok,
        state_token_id=text_tokenizer.encode("<state>")[-1],
        multi_state=args.multi_state,
        **flamingo_kwargs,
    )

    # Freeze all parameters
    model.requires_grad_(False)
    # assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    # # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings

    # model.img_id_embedding.requires_grad_(True)
    # model.perceiver.requires_grad_(True)
    model.mapper.requires_grad_(True)
    model.lang_encoder.requires_grad_(True)
    model.angle_encoder.requires_grad_(True)

    # if r2r_tok:
    #     model.history_encoder.requires_grad_(True)

    # model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
    model.lang_encoder.get_input_embeddings().requires_grad_(True)

    print(
        f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    return model, image_processor, text_tokenizer


def _infer_decoder_layers_attr_name(model):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
    )


__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptneo": "transformer.h",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
}
