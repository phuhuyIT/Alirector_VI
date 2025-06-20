#!/usr/bin/env python
"""
Simple inference script for Vietnamese GEC with ViT5 + LoRA.

Usage:
    python -m src.inference --model_path runs/vit5_distilled --text "Tôi đi học ở trường đại học"
    python -m src.inference --model_path runs/vit5_distilled --file input.txt --output output.txt
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import os


def load_model(model_path: str):
    """Load ViT5 model with LoRA adapters."""
    print(f"Loading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        # Load base model and LoRA adapters
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            "VietAI/vit5-base",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        print("Loaded LoRA model")
    else:
        # Load regular model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Loaded regular model")
    
    model.eval()
    return model, tokenizer


def correct_text(text: str, model, tokenizer, device="cuda", num_beams=4, max_length=192):
    """Correct a single text."""
    # Add instruction prefix
    input_text = f"gec: {text}"
    
    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True
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
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected


def correct_batch(texts, model, tokenizer, device="cuda", batch_size=16, num_beams=4, max_length=192):
    """Correct multiple texts in batches."""
    corrected_texts = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        # Add instruction prefix
        prefixed_batch = [f"gec: {text}" for text in batch]
        
        # Tokenize
        inputs = tokenizer(
            prefixed_batch,
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
        batch_corrected = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        corrected_texts.extend(batch_corrected)
    
    return corrected_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the ViT5 model")
    parser.add_argument("--text", type=str, default=None,
                       help="Single text to correct")
    parser.add_argument("--file", type=str, default=None,
                       help="Input file with texts to correct (one per line)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file to save corrections")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for processing")
    parser.add_argument("--num_beams", type=int, default=4,
                       help="Number of beams for beam search")
    parser.add_argument("--max_length", type=int, default=192,
                       help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Check arguments
    if not args.text and not args.file:
        print("Error: Must provide either --text or --file")
        return
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    
    if args.text:
        # Single text correction
        print(f"Original: {args.text}")
        corrected = correct_text(args.text, model, tokenizer, device, args.num_beams, args.max_length)
        print(f"Corrected: {corrected}")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(corrected + '\n')
            print(f"Saved to: {args.output}")
    
    elif args.file:
        # File processing
        print(f"Processing file: {args.file}")
        
        # Read input file
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(texts)} texts to correct")
        
        # Correct texts
        corrected_texts = correct_batch(texts, model, tokenizer, device, 
                                      args.batch_size, args.num_beams, args.max_length)
        
        # Print results
        for i, (original, corrected) in enumerate(zip(texts, corrected_texts)):
            print(f"\n{i+1}. Original: {original}")
            print(f"   Corrected: {corrected}")
        
        # Save to file if specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                for corrected in corrected_texts:
                    f.write(corrected + '\n')
            print(f"\nSaved {len(corrected_texts)} corrections to: {args.output}")


if __name__ == "__main__":
    main()
