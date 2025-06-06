import fire
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import transformers
from transformers import (
    BartConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    BartForConditionalGeneration,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from datasets import load_dataset
from models.modeling_bart_dropsrc import BartForConditionalGenerationwithDropoutSrc
import wandb

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

def main(
    # model/data params
    model_path: str = 'fnlp/bart-large-chinese',
    data_path: str = "",
    output_dir: str = "",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 32,       
    num_train_epochs: int = 5,
    learning_rate: float = 1e-5,
    val_set_size: int = 1000,
    seed: int = 42,
    warmup_ratio : float = 0.1,
    group_by_length: bool = False,
    max_source_length: int = 128,
    max_target_length: int = 128,
    label_smoothing_factor: float = 0.0,
    logging_steps: int = 5,
    input_reverse: bool = False,
    transformer: bool = False,
    lr_scheduler_type: str = "linear",
    optim: str = "adamw_torch",
    patience: int = 5,
    warmup_steps: int = 2000,
    adam_betas: tuple = (0.9, 0.999),
    dropout: float=0.1,
    src_dropout: float=0.2,
    use_tf32: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "Alirector_Vi",
    wandb_entity: str = "phuhuy02003-university-of-transport-and-communications",
    wandb_api_key: str = "",
):  
    set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    if use_wandb:
        wandb.login(key=wandb_api_key)
        wandb.init(project=wandb_project, entity=wandb_entity)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = local_rank    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    gradient_accumulation_steps = batch_size // micro_batch_size
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(local_rank):
        transformers.utils.logging.set_verbosity_info()

    # tokenizer initialization
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    sep_token = tokenizer.sep_token or tokenizer.eos_token or "</s>"
    if transformer:      # transformer
        dropout=dropout
        activation_function='relu'
        activation_dropout=0.0
        attention_dropout=0.0
        src_dropout=src_dropout
        max_position_embeddings=512
        config = BartConfig.from_pretrained(model_path, dropout=dropout,
                                            activation_function=activation_function,
                                            activation_dropout=activation_dropout,
                                            attention_dropout=attention_dropout,
                                            max_position_embeddings=max_position_embeddings)
        model = BartForConditionalGenerationwithDropoutSrc(config, src_dropout=src_dropout)
    else:
        dropout=dropout
        activation_function='gelu'
        activation_dropout=0.0
        attention_dropout=0.0
        src_dropout=src_dropout
        config = BartConfig.from_pretrained(model_path, dropout=dropout,
                                            activation_function=activation_function,
                                            activation_dropout=activation_dropout,
                                            attention_dropout=attention_dropout)
        model = BartForConditionalGenerationwithDropoutSrc.from_pretrained(model_path, config=config, src_dropout=src_dropout)
        
    model.config.max_length=max_target_length

    # data
    data = load_dataset("json", data_files={"train": data_path})
    datasets = data["train"].train_test_split(
        test_size=val_set_size, shuffle=True, seed=seed
    )
    column_names = datasets["train"].column_names

    def preprocess_function(examples):
        inputs = [f"{src}{sep_token}{pred}" if not input_reverse else f"{pred}{sep_token}{src}" for src, pred in zip(examples['source'], examples['pred'])]
        targets = examples['target']
        model_inputs = tokenizer(inputs,
                                max_length=max_source_length,
                                padding=False,
                                truncation=True,
                                return_token_type_ids=False)

        # Setup the tokenizer for targets
        labels = tokenizer(targets,
                        max_length=max_target_length,
                        padding=False,
                        truncation=True,
                        return_token_type_ids=False)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs



    from accelerate import Accelerator
    accelerator = Accelerator()

    train_dataset = datasets["train"]
    eval_dataset = datasets["test"]

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
        print('================ dataset examples ================')
        print('max length: ', max([len(d) for d in train_dataset['input_ids']]))
        print(tokenizer.batch_decode(train_dataset['input_ids'][:2]))
        print(tokenizer.batch_decode(train_dataset['labels'][:2]))
        print(train_dataset[0])
        print(train_dataset[1])

    eval_steps = 1 / num_train_epochs
    warmup_ratio = 1 / num_train_epochs
    adam_beta1, adam_beta2 = adam_betas
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # warmup_ratio=warmup_ratio,
            warmup_steps=warmup_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            # bf16=True if torch.cuda.is_bf16_supported() else False,
            # fp16=False if torch.cuda.is_bf16_supported() else True,
            fp16=True,
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
            report_to=["tensorboard","wandb"] if use_wandb else "tensorboard",
            label_smoothing_factor=label_smoothing_factor,
            load_best_model_at_end=True,
            tf32=use_tf32,
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
        callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=patience)]
    )

    # Training
    trainer.train()
    trainer.save_model(os.path.join(output_dir, 'best-model'))  # Saves the tokenizer too for easy upload
    
if __name__ == '__main__':
    fire.Fire(main)