import os
import sys
import fire
from typing import List, Any

import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process

from accelerate import Accelerator
import wandb
from peft import LoraConfig, get_peft_model


def main(
    model_path: str = "vinai/bartpho-syllable",
    data_path: str = "",
    output_dir: str = "./align_bartpho_lora_fw",
    wandb_project: str = "alirector_seq2seq",
    wandb_entity: str = "",
    wandb_run_name: str = "",
    wandb_api_key: str = "",
    # flags
    input_reverse: bool = False,  # False -> forward Align (src+pred), True -> reverse Align (pred+src)
    # training hyperparams
    batch_size: int = 64,
    micro_batch_size: int = 16,
    num_train_epochs: int = 3,
    learning_rate: float = 2e-5,
    val_set_size: int = 1000,
    seed: int = 42,
    group_by_length: bool = False,
    max_source_length: int = 128,
    max_target_length: int = 128,
    logging_steps: int = 20,
    lr_scheduler_type: str = "linear",
    optim: str = "adamw_torch",
    warmup_steps: int = 500,
    patience: int = 5,
    # LoRA
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "q_proj,k_proj,v_proj,out_proj",
    flash_attn: bool = True,
    # enable TensorFloat-32 on Ampere+ GPUs
    tf32: bool = False,
):
    """Train alignment model (forward or reverse) with BARTpho + LoRA."""

    set_seed(seed)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    gradient_accumulation_steps = batch_size // micro_batch_size
    if ddp:
        gradient_accumulation_steps //= world_size

    if is_main_process(local_rank):
        transformers.utils.logging.set_verbosity_info()
        wandb.login(key=wandb_api_key)
        wandb.init(
            project=wandb_project,
            entity=wandb_entity if wandb_entity else None,
            name=wandb_run_name if wandb_run_name else None,
            config=locals(),
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    config = AutoConfig.from_pretrained(model_path)
    if flash_attn:
        try:
            config.attn_implementation = "flash_attention_2"
        except Exception:
            if is_main_process(local_rank):
                print("[Warning] flash_attention_2 not supported – fallback.")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    if use_lora:
        targets: List[str] = [m.strip() for m in lora_target_modules.split(",") if m.strip()]
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=targets,
            task_type="SEQ_2_SEQ_LM",
        )
        model = get_peft_model(model, lora_cfg)
        if is_main_process(local_rank):
            model.print_trainable_parameters()

    sep_token = tokenizer.sep_token or tokenizer.eos_token or "</s>"

    if not data_path:
        raise ValueError("--data_path required (json with fields source, pred, target)")

    data_files = {"train": data_path}
    data = load_dataset("json", data_files=data_files)
    datasets = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=seed)
    col_names = datasets["train"].column_names

    def preprocess(examples):
        if input_reverse:
            inputs = [f"{p}{sep_token}{s}" for s, p in zip(examples["source"], examples["pred"])]
        else:
            inputs = [f"{s}{sep_token}{p}" for s, p in zip(examples["source"], examples["pred"])]
        targets = examples["target"]
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=False, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=False, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    accelerator = Accelerator()

    train_ds, eval_ds = datasets["train"], datasets["test"]
    with accelerator.main_process_first():
        train_ds = train_ds.map(preprocess, batched=True, num_proc=os.cpu_count(), remove_columns=col_names, load_from_cache_file=True)
    with accelerator.main_process_first():
        eval_ds = eval_ds.map(preprocess, batched=True, num_proc=os.cpu_count(), remove_columns=col_names, load_from_cache_file=True)

    if local_rank == 0:
        print("Dataset examples:")
        print(tokenizer.batch_decode(train_ds["input_ids"][:2]))
        print(tokenizer.batch_decode(train_ds["labels"][:2]))

    eval_steps = 1 / num_train_epochs

    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
            logging_steps=logging_steps,
            lr_scheduler_type=lr_scheduler_type,
            optim=optim,
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_steps,
            save_steps=eval_steps,
            output_dir=output_dir,
            save_total_limit=3,
            ddp_find_unused_parameters=True if ddp else None,
            group_by_length=group_by_length,
            report_to=["tensorboard","wandb"],
            load_best_model_at_end=True,
            tf32=tf32,
        ),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8),
        callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    trainer.train()

    # save
    if local_rank == 0:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        if use_lora:
            model.save_pretrained(os.path.join(output_dir, "lora_adapter"))


if __name__ == "__main__":
    fire.Fire(main)
