#!/usr/bin/env python
"""
Stage-2 Alirector – train a forward or reverse ALIGNMENT model for ViT5 with LoRA.

* expects a dataset saved by predict.py with columns: input, pred, target
* builds source sentence:  gec: X <extra_id_0> Ŷ   (direction=forward)
                           gec: Ŷ <extra_id_0> X   (direction=reverse)
* works with ViT5 and includes LoRA training
* logs to Weights & Biases
"""

import os, argparse, wandb, torch
from functools import lru_cache
from datasets import load_from_disk, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)
from peft import LoraConfig, get_peft_model, TaskType

# ---------- optional Vietnamese word-segmentation ---------------------------
try:
    from py_vncorenlp import VnCoreNLP      # pip install py_vncorenlp
except ImportError:
    VnCoreNLP = None


# ---------------------------------------------------------------------------
# helpers for VNCoreNLP
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_segmenter(word_segment_save_dir: str):
    if VnCoreNLP is None:
        raise RuntimeError("py_vncorenlp not installed – "
                           "pip install py_vncorenlp and ensure Java ≥8")
    return VnCoreNLP(save_dir=word_segment_save_dir, annotators=["wseg"])


def wseg_sentence(sent: str, word_segment_save_dir: str) -> str:
    """Return sentence with multi-syllable words joined by underscores."""
    seg = get_segmenter(word_segment_save_dir)
    return " ".join(seg.word_segment(sent)[0])


def maybe_segment(texts, seg_needed, args):
    """Segment a list[str] if seg_needed else return original list."""
    if not seg_needed:
        return texts
    seg = get_segmenter(args.word_segment_save_dir)
    # word_segment expects a single string, returns List[List[str]]
    # Process each text individually
    segmented_results = []
    for text in texts:
        segmented_words = seg.word_segment(text)[0]
        segmented_results.append(" ".join(segmented_words))  # Fixed: add spaces
    return segmented_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", type=str, required=True,
                   help="Path to dataset dir (output of predict.py)")
    p.add_argument("--direction", type=str, choices=["forward", "reverse"], 
                   required=True, help="forward: X<>Ŷ, reverse: Ŷ<>X")
    p.add_argument("--model_name_or_path", type=str, default="VietAI/vit5-base")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_source_len", type=int, default=384)
    p.add_argument("--max_target_len", type=int, default=384)
    
    # LoRA parameters
    p.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    # disable LoRA if desired
    p.add_argument("--no_lora", action="store_true", help="Disable LoRA and fine-tune all model parameters")
    
    # training
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=int, default=6)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    
    # wandb
    p.add_argument("--wandb_project", type=str, default="Vi_Alirector_vit5_align")
    p.add_argument("--wandb_entity", type=str, default="phuhuy02003-university-of-transport-and-communications")
    p.add_argument("--wandb_api_key", type=str, default=None)
    
    # optional
    p.add_argument("--word_segment", action="store_true")
    p.add_argument("--word_segment_save_dir", type=str, default="./vncorenlp")
    return p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    import time
    args = build_argparser().parse_args()
    
    # wandb
    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     config=vars(args))
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_dir}...")
    dataset = load_from_disk(args.dataset_dir)
    print(f"Dataset columns: {dataset.column_names}")
    print(f"Dataset size: {len(dataset)}")
    
    # Model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    if not args.no_lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("[INFO] LoRA disabled – training all model parameters.")
    
    # Check if we need word segmentation
    seg_needed = args.word_segment
    print(f"Word segmentation: {'enabled' if seg_needed else 'disabled'}")
    
    # Preprocessing function
    def preprocess_fn(batch):
        sources = []
        targets = []
        
        inputs = batch['input']
        preds = batch['pred'] 
        tgts = batch['target']
        
        # Segment if needed - apply to both source AND predictions
        if seg_needed:
            inputs = maybe_segment(inputs, True, args)
            preds = maybe_segment(preds, True, args)
            tgts = maybe_segment(tgts, True, args)
        
        for inp, pred, tgt in zip(inputs, preds, tgts):
            if args.direction == "forward":
                # Forward: X <extra_id_0> Ŷ -> Y
                source = f"gec: {inp} <extra_id_0> {pred}"
                target = tgt
            else:
                # Reverse: Ŷ <extra_id_0> X -> Y  
                source = f"gec: {pred} <extra_id_0> {inp}"
                target = tgt
                
            sources.append(source)
            targets.append(target)
        
        # Tokenize
        model_inputs = tokenizer(
            sources, 
            max_length=args.max_source_len, 
            truncation=True,
            padding=False
        )
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, 
                max_length=args.max_target_len, 
                truncation=True,
                padding=False
            )
        
        # Replace pad_token_id with -100 in labels (for ignore_index in loss)
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Split dataset
    if "train" in dataset.column_names or hasattr(dataset, 'train'):
        # Already split
        train_ds = dataset["train"] if "train" in dataset else dataset
        if "validation" in dataset:
            eval_ds = dataset["validation"]
        else:
            # Split train 90/10
            split_ds = train_ds.train_test_split(test_size=0.1, seed=42)
            train_ds, eval_ds = split_ds["train"], split_ds["test"]
    else:
        # Split the whole dataset 90/10
        split_ds = dataset.train_test_split(test_size=0.1, seed=42)
        train_ds, eval_ds = split_ds["train"], split_ds["test"]
    
    print(f"Train size: {len(train_ds)}, Eval size: {len(eval_ds)}")
    
    # Apply preprocessing
    train_ds = train_ds.map(preprocess_fn, batched=True, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(preprocess_fn, batched=True, remove_columns=eval_ds.column_names)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        eval_steps=500,
        save_steps=1000,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=f"vit5-align-{args.direction}-{time.strftime('%Y%m%d-%H%M%S')}",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print(f"Starting {args.direction} alignment training...")
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    wandb.finish()
    print(f"Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
