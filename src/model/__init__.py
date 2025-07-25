from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any
import os, torch, logging
from model.probe import ProbedLlamaForCausalLM

logger  = logging.getLogger(__name__)
hf_home = os.getenv("HF_HOME", None)

MODEL_REGISTRY: Dict[str, Any] = {}
def _register_model(cls):
    MODEL_REGISTRY[cls.__name__] = cls

_register_model(AutoModelForCausalLM)
_register_model(ProbedLlamaForCausalLM)

def get_model(model_cfg: DictConfig, *, attn_impl: str | None = None):
    m_args = OmegaConf.to_container(model_cfg.model_args, resolve=True)
    t_args = OmegaConf.to_container(model_cfg.tokenizer_args, resolve=True)

    model_path      = m_args.pop("pretrained_model_name_or_path", None)
    torch_dtype_str = m_args.pop("torch_dtype", None)

    if attn_impl is not None:
        m_args["attn_implementation"] = attn_impl

    if m_args.get("attn_implementation") == "flash_attention_2":
        if torch_dtype_str not in ("float16", "bfloat16"):
            logger.warning(
                "torch_dtype=%r unsupported for flash_attention_2, defaulting to bfloat16",
                torch_dtype_str,
            )
            torch_dtype_str = "bfloat16"
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }.get(torch_dtype_str, torch.float32)

    handler   = model_cfg.get("model_handler", "AutoModelForCausalLM")
    model_cls = MODEL_REGISTRY[handler]
    try:
        model = model_cls.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            **m_args,
            cache_dir=hf_home,
        )
    except Exception as e:
        logger.error("Failed to load model %r: %s", model_path, e)
        raise

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            t_args.pop("pretrained_model_name_or_path"),
            **t_args,
            cache_dir=hf_home,
        )
    except Exception as e:
        logger.error("Failed to load tokenizer: %s", e)
        raise

    # 7) Убеждаемся, что есть eos и pad
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
