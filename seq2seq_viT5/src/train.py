#!/usr/bin/env python
"""
Fine-tune ViT5 for Vietnamese GEC (Stage-1 Alirector) with LoRA.

Usage (Colab GPU):
  python -m src.train \
      --dataset_name bmd1905/vi-error-correction-v2 \
      --output_dir runs/vit5-gec \
      --wandb_project vietgec --wandb_entity myteam --wandb_api_key $KEY
"""

import os, argparse, wandb
import torch
import torch.nn.functional as F
from datasets import load_dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)
from peft import LoraConfig, get_peft_model, TaskType
from functools import lru_cache
try:
    from py_vncorenlp import VnCoreNLP   # noqa: E402
except ImportError:
    VnCoreNLP = None  # will raise later if user requests word segmentation
from typing import List
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # data & model
    p.add_argument("--dataset_name", type=str, default="bmd1905/vi-error-correction-v2")
    p.add_argument("--model_name_or_path", type=str, default="VietAI/vit5-base")
    p.add_argument("--max_source_len", type=int, default=384)
    p.add_argument("--max_target_len", type=int, default=384)
    p.add_argument("--output_dir", type=str, required=True)
    # dataset column names
    p.add_argument("--source_column", type=str, default="incorrect_text", help="Column name for source/incorrect sentences")
    p.add_argument("--target_column", type=str, default="correct_text", help="Column name for target/correct sentences")
    # train hyper-params
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-5)  # Higher LR for ViT5
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=int, default=6)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    # LoRA parameters
    p.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    # wandb
    p.add_argument("--wandb_project", type=str, default="Vi_Alirector_vit5_base")
    p.add_argument("--wandb_entity", type=str, default="phuhuy02003-university-of-transport-and-communications")
    p.add_argument("--wandb_api_key", type=str, default=None)
    p.add_argument("--word_segment", action="store_true",
                   help="Run VNCoreNLP word segmentation before tokenisation "
                        "(required for bartpho-word checkpoints)")
    p.add_argument("--word_segment_save_dir", type=str, default="./vncorenlp")
    # Edit-weighted CE and R-Drop
    p.add_argument("--edit_weight", type=float, default=1.0,
                   help="Weight γ for edit-weighted cross-entropy loss")
    p.add_argument("--rdrop_weight", type=float, default=0.0,
                   help="Weight for R-Drop regularization")
    return p


# ------------- DEFINE helper ---------------------------------------------------

@lru_cache(maxsize=1)
def get_segmenter(word_segment_save_dir: str):
    # Lazy-load VNCoreNLP only once (fork-safe).
    if VnCoreNLP is None:
        raise RuntimeError("py_vncorenlp not installed")
    return VnCoreNLP(save_dir=word_segment_save_dir, annotators=["wseg"])


def segment_batch(texts, args):
    # Segment a list of raw sentences -> list of 'word1 word2' strings.
    seg = get_segmenter(args.word_segment_save_dir)
    
    # Clean inputs - replace None and empty strings with safe placeholder
    clean_texts = []
    for text in texts:
        if text is None or text == "":
            clean_texts.append(".")
        else:
            clean_texts.append(str(text).strip())
    
    print(f"Segmenting {len(clean_texts)} sentences...")
    start_time = time.time()
    
    # Try batch processing first (much faster if available)
    try:
        segmented_lists = seg.word_segment(clean_texts)
        results = [" ".join(words) for words in segmented_lists]
        elapsed = time.time() - start_time
        print(f"Batch segmentation: {len(clean_texts)/elapsed:.1f} sentences/sec")
        return results
    except Exception as e:
        print(f"Batch segmentation failed ({e}), falling back to individual processing...")
        
    # Fallback to individual processing
    results = []
    for text in clean_texts:
        try:
            segmented_words = seg.word_segment(text)[0]
            results.append(" ".join(segmented_words))  # Fixed: add spaces between words
        except Exception:
            results.append(text)  # fallback to original if segmentation fails
    
    elapsed = time.time() - start_time
    print(f"Individual segmentation: {len(clean_texts)/elapsed:.1f} sentences/sec")
    return results


class EditWeightedCrossEntropyTrainer(Seq2SeqTrainer):
    """Custom trainer that applies higher weights to edited tokens in CE loss and R-Drop regularization."""
    
    def __init__(self, edit_weight=1.0, rdrop_weight=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edit_weight = edit_weight
        self.rdrop_weight = rdrop_weight
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute edit-weighted cross entropy loss with optional R-Drop regularization."""
        labels = inputs.get("labels")
        
        # R-Drop: Run forward pass twice with different dropout patterns
        if self.rdrop_weight > 0.0 and model.training:
            # First forward pass
            outputs1 = model(**inputs)
            logits1 = outputs1.get('logits')
            
            # Second forward pass (different dropout)
            outputs2 = model(**inputs)
            logits2 = outputs2.get('logits')
            
            # Use first output for main loss
            outputs = outputs1
        else:
            # Single forward pass
            outputs = model(**inputs)
            logits1 = outputs.get('logits')
        
        logits = outputs.get('logits')
        
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for loss calculation
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Create mask for non-padding tokens
            mask = (shift_labels != -100)
            
            # Apply edit weighting to all non-padding tokens
            if self.edit_weight != 1.0:
                # Compute unreduced cross entropy
                ce_losses = F.cross_entropy(shift_logits, shift_labels, reduction='none')
                
                # Apply edit weight to non-padding positions
                weighted_losses = torch.where(mask, 
                                             ce_losses * self.edit_weight, 
                                             ce_losses)
                
                # Average over valid positions
                loss_ce = weighted_losses[mask].mean()
            else:
                # Standard cross entropy with padding mask
                loss_ce = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        else:
            loss_ce = torch.tensor(0.0, device=logits.device)
        
        # R-Drop regularization
        loss_rdrop = torch.tensor(0.0, device=logits.device)
        if self.rdrop_weight > 0.0 and model.training and labels is not None:
            # Compute KL divergence between two outputs
            shift_logits1 = logits1[..., :-1, :].contiguous().view(-1, logits1.size(-1))
            shift_logits2 = logits2[..., :-1, :].contiguous().view(-1, logits2.size(-1))
            
            # Apply mask to both logits
            valid_logits1 = shift_logits1[mask]
            valid_logits2 = shift_logits2[mask]
            
            if valid_logits1.size(0) > 0:
                # Symmetric KL divergence
                kl1 = F.kl_div(F.log_softmax(valid_logits1, dim=-1), 
                              F.softmax(valid_logits2, dim=-1), 
                              reduction='batchmean')
                kl2 = F.kl_div(F.log_softmax(valid_logits2, dim=-1), 
                              F.softmax(valid_logits1, dim=-1), 
                              reduction='batchmean')
                loss_rdrop = 0.5 * (kl1 + kl2)
        
        # Total loss
        loss = loss_ce + self.rdrop_weight * loss_rdrop
        
        # Log individual loss components
        if self.rdrop_weight > 0.0 and model.training:
            self.log({"train/loss_ce": loss_ce.item(), 
                     "train/loss_rdrop": loss_rdrop.item(),
                     "train/loss_total": loss.item()})
        
        return (loss, outputs) if return_outputs else loss


def main():
    import time
    args = build_argparser().parse_args()
    
    # wandb
    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     config=vars(args))
    
    # model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Add prefix token for T5-style instruction
    # ViT5 uses <extra_id_X> tokens, we'll use a simple instruction prefix
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,  # Use bf16 as requested
        device_map="auto"
    )
    
    # Setup LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo"]  # ViT5 attention modules
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    if os.path.isdir(args.dataset_name):
        ds = load_from_disk(args.dataset_name)
    else:
        ds = load_dataset(args.dataset_name)
    
    # Preprocessing function with instruction prefix
    def preprocess_fn(batch):
        sources = []
        targets = []
        
        source_batch = batch[args.source_column]
        target_batch = batch[args.target_column]
        for src, tgt in zip(source_batch, target_batch):
            # Add "gec: " prefix to source as instruction
            source = f"gec: {src}"
            
            # Apply word segmentation if needed
            if args.word_segment:
                source = segment_batch([source], args)[0]
                tgt = segment_batch([tgt], args)[0]
            
            sources.append(source)
            targets.append(tgt)
        
        # Tokenize
        model_inputs = tokenizer(
            sources, max_length=args.max_source_len, truncation=True
        )
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=args.max_target_len, truncation=True
            )
        
        # Replace pad token id with -100 for labels
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Apply preprocessing
    train_ds = ds["train"].map(preprocess_fn, batched=True, remove_columns=ds["train"].column_names)
    eval_ds = ds["validation"].map(preprocess_fn, batched=True, remove_columns=ds["validation"].column_names)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,  # Use bf16 as requested
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
        run_name=f"vit5-gec-{time.strftime('%Y%m%d-%H%M%S')}",
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
    trainer = EditWeightedCrossEntropyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        edit_weight=args.edit_weight,
        rdrop_weight=args.rdrop_weight,
    )
    
    # Train
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Log final metrics
    wandb.finish()


if __name__ == "__main__":
    main()
