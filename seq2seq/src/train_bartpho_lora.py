import os
import sys
import fire
import json
from typing import List, Union

import torch
import transformers
from datasets import load_dataset, Dataset
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

# LoRA / PEFT
from peft import LoraConfig, get_peft_model

# Optional: accelerate for multi-GPU & mixed precision
from accelerate import Accelerator
import wandb

def main(
    # model / data params
    model_path: str = "vinai/bartpho-syllable",
    data_path: str = "",
    output_dir: str = "./bartpho_lora_outputs",
    wandb_project: str = "alirector_seq2seq",
    wandb_entity: str = "",
    wandb_run_name: str = "",
    wandb_api_key: str = "",
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
    # LoRA params
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "q_proj,k_proj,v_proj,out_proj",
    # flash-attention
    flash_attn: bool = False,
    # enable TensorFloat-32 on Ampere+ GPUs
    tf32: bool = False,
):
    """Simple finetuning script for Vietnamese Grammar Error Correction based on BARTpho + LoRA.

    Example usage:
        python train_bartpho_lora.py \
            --data_path data/train.jsonl \
            --output_dir outputs/bartpho_lora \
            --num_train_epochs 5
    """
    # ===== env & seed =====
    set_seed(seed)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    gradient_accumulation_steps = batch_size // micro_batch_size
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # ===== logging =====
    if is_main_process(local_rank):
        transformers.utils.logging.set_verbosity_info()
        print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
        wandb.login(key=wandb_api_key)
        wandb.init(
            project=wandb_project,
            entity=wandb_entity if wandb_entity else None,
            name=wandb_run_name if wandb_run_name else None,
            config=locals(),
        )

    # ===== tokenizer & model =====
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    config = AutoConfig.from_pretrained(model_path)
    # if flash_attn:
    #     try:
    #         # Newer versions of HF allow this attribute; silently ignore if BARTpho does not support.
    #         config.attn_implementation = "flash_attention_2"
    #     except Exception as e:
    #         if is_main_process(local_rank):
    #             print("[Warning] flash_attention_2 not supported for this model – fallback to default attention.")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    if use_lora:
        target_modules: List[str]
        if isinstance(lora_target_modules, str):
            target_modules = [m.strip() for m in lora_target_modules.split(",") if m.strip()]
        else:
            target_modules = lora_target_modules  # type: ignore
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type="SEQ_2_SEQ_LM",
        )
        model = get_peft_model(model, lora_config)
        if is_main_process(local_rank):
            model.print_trainable_parameters()

    # make sure model can generate reasonably long outputs
    model.config.max_length = max_target_length

    # ===== dataset =====
    if not data_path:
        raise ValueError("--data_path must be provided and point to a JSONL file with fields 'source', 'pred', 'target'.")
    # manual load JSONL to avoid LocalFileSystem caching issues
    with open(data_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    raw_ds = Dataset.from_list(records)
    datasets = raw_ds.train_test_split(test_size=val_set_size, shuffle=True, seed=seed)
    column_names = raw_ds.column_names

    sep_token = tokenizer.sep_token or tokenizer.eos_token or "</s>"

    def preprocess_function(examples):
        inputs = [f"{src}{sep_token}{pred}" for src, pred in zip(examples["source"], examples["pred"])]
        targets = examples["target"]
        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            padding=False,
            truncation=True,
            return_token_type_ids=False,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_target_length,
                padding=False,
                truncation=True,
                return_token_type_ids=False,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    accelerator = Accelerator()

    train_dataset, eval_dataset = datasets["train"], datasets["test"]

    with accelerator.main_process_first():
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=os.cpu_count(),
            remove_columns=column_names,
            load_from_cache_file=True,
        )
    with accelerator.main_process_first():
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=os.cpu_count(),
            remove_columns=column_names,
            load_from_cache_file=True,
        )

    if local_rank == 0:
        print("================ Dataset Examples ================")
        print("Max length:", max(len(d) for d in train_dataset["input_ids"]))
        print(tokenizer.batch_decode(train_dataset["input_ids"][:2]))
        print(tokenizer.batch_decode(train_dataset["labels"][:2]))

    # Using per-epoch evaluation/saving instead of float eval_steps

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
            eval_strategy="epoch",
            save_strategy="epoch",
            output_dir=output_dir,
            save_total_limit=3,
            ddp_find_unused_parameters=True if ddp else None,
            group_by_length=group_by_length,
            report_to=["tensorboard","wandb"],
            load_best_model_at_end=True,
            tf32=tf32,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        ),
        callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    trainer.train()

    if local_rank == 0:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        if use_lora:
            # Save LoRA adapter weights separately for lightweight deployment.
            model.save_pretrained(os.path.join(output_dir, "lora_adapter"))


if __name__ == "__main__":
    fire.Fire(main)
