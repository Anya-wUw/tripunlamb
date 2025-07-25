import hydra
from omegaconf import DictConfig
import torch
import logging

from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything, compute_kl_divergence
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)

def resolve_target_modules(requested, model):
    # filter LoRA target modules by presence in model
    modules = {n for n, _ in model.named_modules()}
    valid = [m for m in requested
             if any(n.endswith(f".{m}") or n == m for n in modules)]
    missing = set(requested) - set(valid)
    if missing:
        logger.warning("Skipped missing LoRA modules: %s", missing)
    if not valid:
        raise ValueError("No valid LoRA modules found.")
    return valid

def load_base_model(cfg):
    # try preferred then fallback attention
    backends = [cfg.model_args.get("attn_implementation", "flash_attention_2"), "sdpa"]
    errors = []
    for impl in backends:
        try:
            model, tokenizer = get_model(cfg, attn_impl=impl)
            model.gradient_checkpointing_enable()
            logger.info("Loaded model with attn_impl=%s", impl)
            return model, tokenizer
        except Exception as e:
            errors.append((impl, str(e)))
            logger.warning("Backend %s failed: %s", impl, e)
    raise RuntimeError(f"All attention backends failed: {errors}")

@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.trainer.args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model and tokenizer
    model, tokenizer = load_base_model(cfg.model)
    model.to(device)
    model.config.use_cache = False

    # frozen copy for KL
    ref_model, _ = load_base_model(cfg.model)
    ref_model.to(device)
    ref_model.requires_grad_(False)

    # apply LoRA if enabled
    if cfg.peft.lora.enabled:
        targets = resolve_target_modules(cfg.peft.lora.target_modules, model)
        l = cfg.peft.lora
        peft_cfg = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=l.r,
            lora_alpha=l.alpha,
            lora_dropout=l.dropout,
            target_modules=targets,
            bias="lora_only",
        )
        model = get_peft_model(model, peft_cfg)
        logger.info("Applied LoRA adapter")

    # log sizes
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model size: total=%.2fM, trainable=%.2fM", total/1e6, trainable/1e6)

    # prepare data
    mode = cfg.get("mode", "train")
    data = get_data(cfg.data, mode=mode, tokenizer=tokenizer, template_args=cfg.model.template_args)
    collator = get_collators(cfg.collator, tokenizer=tokenizer)

    # evaluators
    evaluators = None
    if cfg.get("eval"):
        evaluators = get_evaluators(cfg.eval, template_args=cfg.model.template_args,
                                    model=model, tokenizer=tokenizer)

    # trainer setup
    trainer, targs = load_trainer(
        trainer_cfg   = cfg.trainer,
        model         = model,
        train_dataset = data.get("train"),
        eval_dataset  = data.get("eval"),
        tokenizer     = tokenizer,
        data_collator = collator,
        evaluators    = evaluators,
        template_args = cfg.model.template_args,
    )
    trainer.model.config.use_cache = False
    trainer.args.predict_with_generate = True
    gen = getattr(cfg.trainer, "generation_kwargs", {})
    trainer.args.generation_max_new_tokens = gen.get("max_new_tokens", 4)
    trainer.args.generation_num_beams      = gen.get("num_beams", 1)

    # KL-regularized loss
    base_loss = trainer.compute_loss
    def compute_with_kl(model_, inputs, return_outputs=False):
        ce_loss, outputs = base_loss(model_, inputs, return_outputs=True)
        if "forget" in inputs:
            f = inputs["forget"]
            kl_inputs = {"input_ids": f["input_ids"],
                         "attention_mask": f["attention_mask"],
                         "labels": f["labels"]}
        else:
            kl_inputs = inputs
        kl, _ = compute_kl_divergence(model_, ref_model, kl_inputs)
        loss = ce_loss + 0.02 * kl
        return (loss, outputs) if return_outputs else loss
    trainer.compute_loss = compute_with_kl

    # train, merge, save
    if targs.do_train:
        trainer.train()
        if cfg.peft.lora.enabled:
            model = model.merge_and_unload()
            setattr(model, "_hf_peft_config_loaded", False)
            logger.info("Merged LoRA into base model")
        model.save_pretrained(targs.output_dir)
        tokenizer.save_pretrained(targs.output_dir)

    # evaluate
    if targs.do_eval:
        trainer.evaluate(metric_key_prefix="eval")

if __name__ == "__main__":
    main()
