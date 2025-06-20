#!/usr/bin/env python
"""
Evaluate Vietnamese GEC model performance using ViT5 with LoRA support.

Supports BLEU, ROUGE, exact match, and other metrics.

Usage:
    python -m src.evaluate_gec \
        --model_path runs/vit5-gec \
        --dataset_name bmd1905/vi-error-correction-v2 \
        --split test \
        --output_file results/evaluation.json
"""

import argparse
import json
import os
import time
from typing import List, Dict, Any

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from tqdm import tqdm
import numpy as np

# Import metrics
try:
    from datasets import load_metric
    bleu_metric = load_metric("bleu")
    rouge_metric = load_metric("rouge")
except:
    # Fallback for newer transformers versions
    from evaluate import load
    bleu_metric = load("bleu")
    rouge_metric = load("rouge")

try:
    from py_vncorenlp import VnCoreNLP
except ImportError:
    VnCoreNLP = None


def compute_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute BLEU scores."""
    # Tokenize for BLEU computation
    pred_tokens = [pred.split() for pred in predictions]
    ref_tokens = [[ref.split()] for ref in references]
    
    try:
        result = bleu_metric.compute(predictions=pred_tokens, references=ref_tokens)
        return {
            "bleu": result["bleu"],
            "bleu_1": result["precisions"][0] if len(result["precisions"]) > 0 else 0.0,
            "bleu_2": result["precisions"][1] if len(result["precisions"]) > 1 else 0.0,
            "bleu_3": result["precisions"][2] if len(result["precisions"]) > 2 else 0.0,
            "bleu_4": result["precisions"][3] if len(result["precisions"]) > 3 else 0.0,
        }
    except:
        # Fallback computation
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoothie = SmoothingFunction().method1
        
        bleu_scores = []
        for pred, ref in zip(pred_tokens, ref_tokens):
            score = sentence_bleu(ref, pred, smoothing_function=smoothie)
            bleu_scores.append(score)
        
        return {"bleu": np.mean(bleu_scores)}


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores."""
    try:
        result = rouge_metric.compute(predictions=predictions, references=references)
        return {
            "rouge1": result["rouge1"].mid.fmeasure,
            "rouge2": result["rouge2"].mid.fmeasure,
            "rougeL": result["rougeL"].mid.fmeasure,
        }
    except:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def compute_exact_match(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute exact match accuracy."""
    exact_matches = [pred.strip() == ref.strip() for pred, ref in zip(predictions, references)]
    return {"exact_match": np.mean(exact_matches)}


def compute_character_level_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute character-level edit distance metrics."""
    from difflib import SequenceMatcher
    
    similarities = []
    edit_distances = []
    
    for pred, ref in zip(predictions, references):
        # Character-level similarity
        matcher = SequenceMatcher(None, pred, ref)
        similarity = matcher.ratio()
        similarities.append(similarity)
        
        # Simple edit distance approximation
        edit_distance = len(ref) + len(pred) - 2 * len(pred) * similarity
        edit_distances.append(edit_distance)
    
    return {
        "char_similarity": np.mean(similarities),
        "avg_edit_distance": np.mean(edit_distances),
    }


def segment_text_if_needed(texts: List[str], word_segment: bool, word_segment_save_dir: str) -> List[str]:
    """Apply word segmentation if needed."""
    if not word_segment or VnCoreNLP is None:
        return texts
    
    segmenter = VnCoreNLP(save_dir=word_segment_save_dir, annotators=["wseg"])
    segmented_texts = []
    
    for text in texts:
        try:
            words = segmenter.word_segment(text)[0]
            segmented_texts.append(" ".join(words))
        except:
            segmented_texts.append(text)  # fallback
    
    return segmented_texts


def generate_predictions(
    model, 
    tokenizer, 
    texts: List[str], 
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 192,
    num_beams: int = 4,
    word_segment: bool = False,
    word_segment_save_dir: str = "./vncorenlp"
) -> List[str]:
    """Generate predictions using the model."""
    
    # Add instruction prefix and segment if needed
    prefixed_texts = [f"gec: {text}" for text in texts]
    if word_segment:
        prefixed_texts = segment_text_if_needed(prefixed_texts, True, word_segment_save_dir)
    
    predictions = []
    
    for i in tqdm(range(0, len(prefixed_texts), batch_size), desc="Generating predictions"):
        batch_texts = prefixed_texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_beams=num_beams,
                do_sample=False,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(batch_predictions)
    
    return predictions


def build_argparser():
    parser = argparse.ArgumentParser()
    
    # Model and data
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the ViT5 model (with or without LoRA)")
    parser.add_argument("--dataset_name", type=str, default="bmd1905/vi-error-correction-v2")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--output_file", type=str, default="evaluation_results.json")
    
    # Generation parameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--num_beams", type=int, default=4)
    
    # Vietnamese-specific
    parser.add_argument("--word_segment", action="store_true",
                       help="Apply VNCoreNLP word segmentation")
    parser.add_argument("--word_segment_save_dir", type=str, default="./vncorenlp")
    
    # Evaluation options
    parser.add_argument("--compute_all_metrics", action="store_true",
                       help="Compute all available metrics (slower)")
    parser.add_argument("--subset_size", type=int, default=None,
                       help="Evaluate on a subset of the data")
    
    return parser


def main():
    args = build_argparser().parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    try:
        # Check if this is a LoRA model
        if os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
            # Load base model and LoRA adapters
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                "VietAI/vit5-base",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            model = PeftModel.from_pretrained(base_model, args.model_path)
            print("Loaded LoRA model")
        else:
            # Load regular model
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("Loaded regular model")
    except Exception as e:
        print(f"Failed to load with fp16, trying without: {e}")
        if os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
            base_model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")
            model = PeftModel.from_pretrained(base_model, args.model_path)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
        model.to(device)
    
    model.eval()
    
    # Load dataset
    print(f"Loading dataset {args.dataset_name}...")
    if os.path.isdir(args.dataset_name):
        dataset = load_from_disk(args.dataset_name)
    else:
        dataset = load_dataset(args.dataset_name)
    
    # Select split
    if hasattr(dataset, 'keys') and args.split in dataset:
        eval_dataset = dataset[args.split]
    else:
        eval_dataset = dataset
    
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Use subset if specified
    if args.subset_size is not None:
        eval_dataset = eval_dataset.select(range(min(args.subset_size, len(eval_dataset))))
        print(f"Using subset of size: {len(eval_dataset)}")
    
    # Extract inputs and references
    inputs = eval_dataset['input']
    references = eval_dataset['target']
    
    # Generate predictions
    print("Generating predictions...")
    start_time = time.time()
    
    predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        texts=inputs,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_beams=args.num_beams,
        word_segment=args.word_segment,
        word_segment_save_dir=args.word_segment_save_dir
    )
    
    generation_time = time.time() - start_time
    print(f"Generated {len(predictions)} predictions in {generation_time:.2f}s")
    
    # Compute metrics
    print("Computing metrics...")
    
    # Basic metrics
    bleu_scores = compute_bleu(predictions, references)
    exact_match_scores = compute_exact_match(predictions, references)
    
    metrics = {
        **bleu_scores,
        **exact_match_scores,
        "num_samples": len(predictions),
        "generation_time": generation_time,
        "samples_per_second": len(predictions) / generation_time,
    }
    
    # Additional metrics if requested
    if args.compute_all_metrics:
        rouge_scores = compute_rouge(predictions, references)
        char_metrics = compute_character_level_metrics(predictions, references)
        metrics.update(rouge_scores)
        metrics.update(char_metrics)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for metric_name, score in metrics.items():
        if isinstance(score, float):
            print(f"{metric_name}: {score:.4f}")
        else:
            print(f"{metric_name}: {score}")
    
    # Save detailed results
    detailed_results = {
        "metrics": metrics,
        "args": vars(args),
        "examples": []
    }
    
    # Add some examples
    num_examples = min(10, len(predictions))
    for i in range(num_examples):
        detailed_results["examples"].append({
            "input": inputs[i],
            "prediction": predictions[i],
            "reference": references[i],
            "exact_match": predictions[i].strip() == references[i].strip()
        })
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {args.output_file}")
    
    # Print some examples
    print(f"\nSample predictions:")
    print("-" * 50)
    for i in range(min(3, len(predictions))):
        print(f"Input: {inputs[i]}")
        print(f"Prediction: {predictions[i]}")
        print(f"Reference: {references[i]}")
        print(f"Exact Match: {predictions[i].strip() == references[i].strip()}")
        print("-" * 30)


if __name__ == "__main__":
    main()
