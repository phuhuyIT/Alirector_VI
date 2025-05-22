import os
import sys
import fire
from typing import List, Optional, Any, Dict, Tuple

import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process, nested_detach
from transformers.trainer import logger

from accelerate import Accelerator
import wandb
from peft import LoraConfig, get_peft_model, PeftModel

from models.modeling_alignment_distill_bart import AlignmentDistillBART


def load_model_with_lora(base_path: str, lora_path: str | None, flash_attn: bool = True, trainable: bool = False):
    config = AutoConfig.from_pretrained(base_path)
    if flash_attn:
        try:
            config.attn_implementation = "flash_attention_2"
        except Exception:
            pass
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_path, config=config, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    if lora_path and os.path.exists(lora_path):
        base_model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=trainable)
    return base_model


class DistillTrainer(Seq2SeqTrainer):
    """Custom Trainer to correctly save LoRA + base model"""

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # save the composite model (includes peft adapters if any)
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

    # override prediction_step identical to original script for proper eval
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs["lm_loss"].mean().detach()
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1:]
        if prediction_loss_only:
            return (loss, None, None)
        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
        return (loss, logits, None)


def main(
    # paths
    cor_base_path: str = "vinai/bartpho-syllable",
    cor_lora_path: str = "",  # LoRA adapter directory trained in stage 1
    align_fw_base_path: str = "vinai/bartpho-syllable",
    align_fw_lora_path: str = "",  # forward align adapter
    align_rev_base_path: str = "vinai/bartpho-syllable",
    align_rev_lora_path: str = "",  # reverse align adapter
    data_path: str = "",
    output_dir: str = "./distill_bartpho_lora",
    # hyperparams
    batch_size: int = 32,
    micro_batch_size: int = 8,
    num_train_epochs: int = 3,
    learning_rate: float = 2e-5,
    val_set_size: int = 100,
    seed: int = 42,
    group_by_length: bool = False,
    max_cor_length: int = 128,
    max_align_length: int = 128,
    max_target_length: int = 128,
    logging_steps: int = 20,
    warmup_steps: int = 500,
    patience: int = 5,
    # distill params
    kl_loss_weight: float = 0.1,
    alpha: float = 0.5,
    distill_way: str = "average_loss",
    kl_loss_type: str = "forward-kl",
    flash_attn: bool = True,
    # enable TensorFloat-32 on Ampere+ GPUs
    tf32: bool = False,
    # wandb
    wandb_project: str = "alirector_seq2seq",
    wandb_entity: str = "",
    wandb_run_name: str = "",
):
    """Stage-3 Distillation: fine-tune correction model with teacher align models (both directions)."""

    set_seed(seed)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    grad_accum = batch_size // micro_batch_size
    if ddp:
        grad_accum //= world_size

    if is_main_process(local_rank):
        transformers.utils.logging.set_verbosity_info()
        wandb.login()
        wandb.init(
            project=wandb_project,
            entity=wandb_entity if wandb_entity else None,
            name=wandb_run_name if wandb_run_name else None,
            config=locals(),
        )

    tokenizer = AutoTokenizer.from_pretrained(cor_base_path, use_fast=False)
    tokenizer.model_input_names = [
        "input_ids",
        "attention_mask",
        "align_input_ids",
        "align_attention_mask",
    ]

    # === Load models ===
    cor_model = load_model_with_lora(cor_base_path, cor_lora_path, flash_attn, trainable=True)
    align_fw_model = None
    align_rev_model = None
    if kl_loss_weight > 0:
        align_fw_model = load_model_with_lora(align_fw_base_path, align_fw_lora_path, flash_attn, trainable=False)
        align_rev_model = load_model_with_lora(align_rev_base_path, align_rev_lora_path, flash_attn, trainable=False)

    model = AlignmentDistillBART(
        cor_bart=cor_model,
        align_bart=align_fw_model,
        align_bart_reverse=align_rev_model,
        kl_loss_weight=kl_loss_weight,
        kl_loss_type=kl_loss_type,
        distill_way=distill_way,
        alpha=alpha,
    )

    # ===== dataset =====
    if not data_path:
        raise ValueError("--data_path required")
    data_files = {"train": data_path}
    data = load_dataset("json", data_files=data_files)
    datasets = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=seed)
    col_names = datasets["train"].column_names

    sep_token = tokenizer.sep_token or tokenizer.eos_token or "</s>"

    def preprocess(batch):
        inputs = [f"{s}{sep_token}{p}" for s, p in zip(batch["source"], batch["pred"])]
        align_inputs = inputs  # forward
        align_rev_inputs = [f"{p}{sep_token}{s}" for s, p in zip(batch["source"], batch["pred"])]
        model_inputs = tokenizer(inputs, max_length=max_cor_length, padding=False, truncation=True)
        align_ids = tokenizer(align_inputs, max_length=max_align_length, padding=False, truncation=True)
        align_rev_ids = tokenizer(align_rev_inputs, max_length=max_align_length, padding=False, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(batch["target"], max_length=max_target_length, padding=False, truncation=True)
        model_inputs.update(
            {
                "align_input_ids": align_ids["input_ids"],
                "align_attention_mask": align_ids["attention_mask"],
                "align_reverse_input_ids": align_rev_ids["input_ids"],
                "align_reverse_attention_mask": align_rev_ids["attention_mask"],
                "labels": labels["input_ids"],
            }
        )
        return model_inputs

    accelerator = Accelerator()
    train_ds, eval_ds = datasets["train"], datasets["test"]
    with accelerator.main_process_first():
        train_ds = train_ds.map(preprocess, batched=True, num_proc=os.cpu_count(), remove_columns=col_names, load_from_cache_file=True)
    with accelerator.main_process_first():
        eval_ds = eval_ds.map(preprocess, batched=True, num_proc=os.cpu_count(), remove_columns=col_names, load_from_cache_file=True)

    eval_steps = 1 / num_train_epochs

    trainer = DistillTrainer(
        model=model,
        tokenizer=tokenizer,
        args=Seq2SeqTrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=warmup_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
            logging_steps=logging_steps,
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_steps,
            save_steps=eval_steps,
            output_dir=output_dir,
            save_total_limit=3,
            ddp_find_unused_parameters=True if ddp else None,
            group_by_length=group_by_length,
            report_to=["tensorboard", "wandb"],
            optim="adamw_torch",
            tf32=tf32,
        ),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=lambda x: x,  # we already padded in preprocess for variable fields
    )

    trainer.train()

    if local_rank == 0:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        # save LoRA adapter of correction model separately
        if isinstance(cor_model, PeftModel):
            cor_model.save_pretrained(os.path.join(output_dir, "lora_adapter"))


if __name__ == "__main__":
    fire.Fire(main)
