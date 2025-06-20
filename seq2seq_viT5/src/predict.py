#!/usr/bin/env python
"""
Generate draft corrections Ŷ for every X in a dataset (Stage-2 Alirector) for ViT5 with LoRA.
OPTIMIZED VERSION for faster GPU inference on L4/A100 GPUs.

Example (Colab):

    python -m src.predict \
        --model_path runs/vit5-gec \
        --dataset_name bmd1905/vi-error-correction-v2 \
        --split train \
        --output_dir data/stage2/train_pred \
        --fp16 \
        --batch_size 32 \
        --use_amp \
        --torch_compile
"""
import argparse, os, torch, json, time
import wandb
from functools import lru_cache
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import gc
from torch.cuda.amp import autocast
import logging

# Set up logging for performance monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from py_vncorenlp import VnCoreNLP   # optional
except ImportError:
    VnCoreNLP = None


# --------------------------------------------------------------------------
# GPU Optimization Settings
# --------------------------------------------------------------------------
def optimize_gpu_settings():
    """Optimize GPU settings for better performance"""
    if torch.cuda.is_available():
        # Enable TensorFloat-32 on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN benchmarking for consistent input sizes
        torch.backends.cudnn.benchmark = True
        
        # Optimize memory allocation
        torch.cuda.empty_cache()
        logger.info("GPU optimization settings applied")


def clear_gpu_cache():
    """Clear GPU cache to free memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


# --------------------------------------------------------------------------
# Dynamic Batching for Better GPU Utilization
# --------------------------------------------------------------------------
def get_optimal_batch_size(tokenizer, model, device, base_batch_size=16, max_length=192):
    """Dynamically determine optimal batch size based on available GPU memory"""
    if not torch.cuda.is_available():
        return base_batch_size
    
    # Get available GPU memory
    gpu_memory = torch.cuda.get_device_properties(device).total_memory
    available_memory = gpu_memory - torch.cuda.memory_allocated(device)
    
    # Estimate memory per sample (rough heuristic)
    estimated_memory_per_sample = max_length * 1024 * 4  # rough estimate
    
    # Calculate optimal batch size with safety margin
    optimal_batch_size = int(available_memory * 0.8 / estimated_memory_per_sample)
    optimal_batch_size = max(1, min(optimal_batch_size, base_batch_size * 2))
    
    logger.info(f"Optimal batch size: {optimal_batch_size} (base: {base_batch_size})")
    return optimal_batch_size


# --------------------------------------------------------------------------
# Word-segmentation helpers (only needed for bartpho-word checkpoints)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_segmenter(word_segment_save_dir: str):
    if VnCoreNLP is None:
        raise RuntimeError("py_vncorenlp not installed")
    return VnCoreNLP(save_dir=word_segment_save_dir, annotators=["wseg"])


def segment_sentences(batch, args):
    """Segment Vietnamese sentences if word_segment is enabled."""
    if not args.word_segment:
        return batch
    
    seg = get_segmenter(args.word_segment_save_dir)
    segmented = []
    for sent in batch:
        try:
            words = seg.word_segment(sent)[0]
            segmented.append(" ".join(words))  # Fixed: add spaces between words
        except:
            segmented.append(sent)  # fallback
    return segmented


# --------------------------------------------------------------------------
# CLI with additional optimization options
# --------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to the ViT5 model (with or without LoRA)")
    p.add_argument("--dataset_name", type=str, default="bmd1905/vi-error-correction-v2")
    p.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=192)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=192)
    
    # Performance optimization options
    p.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    p.add_argument("--use_amp", action="store_true", help="Use Automatic Mixed Precision")
    p.add_argument("--torch_compile", action="store_true", help="Use torch.compile for faster inference")
    p.add_argument("--dynamic_batching", type=bool, default=True, help="Auto-determine optimal batch size")
    p.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    
    # Vietnamese-specific
    p.add_argument("--word_segment", action="store_true",
                   help="Apply VNCoreNLP word segmentation")
    p.add_argument("--word_segment_save_dir", type=str, default="./vncorenlp")
    
    # Dataset sampling
    p.add_argument("--subset_ratio", type=float, default=1.0,
                   help="Use only a subset of the dataset (0.0-1.0)")
    
    # Wandb
    p.add_argument("--wandb_project", type=str, default="Vi_Alirector_vit5_predict")
    p.add_argument("--wandb_entity", type=str, default="phuhuy02003-university-of-transport-and-communications")
    p.add_argument("--wandb_api_key", type=str, default=None)
    
    return p


# --------------------------------------------------------------------------
# Optimized Generation Function
# --------------------------------------------------------------------------
def create_optimized_generate_function(model, tokenizer, device, args, use_amp=False):
    """Create an optimized generation function with various performance improvements"""
    
    def generate_batch(input_texts):
        # Add instruction prefix to all inputs
        prefixed_texts = [f"gec: {text}" for text in input_texts]
        
        # Apply word segmentation if needed
        if args.word_segment:
            prefixed_texts = segment_sentences(prefixed_texts, args)
        
        # Tokenize
        inputs = tokenizer(
            prefixed_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length
        ).to(device, non_blocking=True)
        
        # Generate with optimizations
        with torch.no_grad():
            if use_amp:
                with autocast():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=args.num_beams,
                        do_sample=False,
                        early_stopping=True,
                        use_cache=True,  # Enable KV caching
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    do_sample=False,
                    early_stopping=True,
                    use_cache=True,  # Enable KV caching
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        
        # Decode outputs
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return predictions
    
    return generate_batch


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    args = build_argparser().parse_args()
    
    # Initialize wandb
    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     config=vars(args))
    
    # Optimize GPU settings
    optimize_gpu_settings()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
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
                torch_dtype=torch.bfloat16 if not args.fp16 else torch.float16,
                low_cpu_mem_usage=True
            )
            model = PeftModel.from_pretrained(base_model, args.model_path)
            logger.info("Loaded LoRA model")
        else:
            # Load regular model
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16 if not args.fp16 else torch.float16,
                low_cpu_mem_usage=True
            )
            logger.info("Loaded regular model")
    except Exception as e:
        logger.warning(f"Failed to load with specified dtype ({e}), trying without dtype specification...")
        if os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
            base_model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")
            model = PeftModel.from_pretrained(base_model, args.model_path)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Apply torch.compile for PyTorch 2.0+
    if args.torch_compile and hasattr(torch, 'compile'):
        logger.info("Applying torch.compile optimization...")
        model = torch.compile(model)
    
    # Load dataset
    print(f"Loading dataset {args.dataset_name}...")
    if os.path.isdir(args.dataset_name):
        dataset = load_from_disk(args.dataset_name)
    else:
        dataset = load_dataset(args.dataset_name)
    
    # Select split
    if isinstance(dataset, DatasetDict):
        if args.split in dataset:
            dataset = dataset[args.split]
        else:
            print(f"Split '{args.split}' not found. Available splits: {list(dataset.keys())}")
            dataset = dataset[list(dataset.keys())[0]]  # Use first available split
    
    print(f"Dataset size: {len(dataset)}")
    
    # Apply subset sampling if requested
    if args.subset_ratio < 1.0:
        original_size = len(dataset)
        subset_size = int(original_size * args.subset_ratio)
        dataset = dataset.shuffle(seed=42).select(range(subset_size))
        print(f"Dataset reduced from {original_size} to {len(dataset)} examples")
    
    # Dynamic batch size optimization
    if args.dynamic_batching:
        optimal_batch_size = get_optimal_batch_size(tokenizer, model, device, args.batch_size, args.max_length)
        batch_size = optimal_batch_size
    else:
        batch_size = args.batch_size
    
    print(f"Using batch size: {batch_size}")
    
    # Create optimized generation function
    generate_batch = create_optimized_generate_function(model, tokenizer, device, args, args.use_amp)
    
    # Generate predictions
    print("Generating predictions...")
    predictions = []
    inputs = []
    targets = []
    
    start_time = time.time()
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
        batch_end = min(i + batch_size, len(dataset))
        batch = dataset[i:batch_end]
        
        # Extract input texts
        if isinstance(batch['input'], list):
            batch_inputs = batch['input']
        else:
            batch_inputs = [batch['input']]
        
        # Generate predictions
        batch_predictions = generate_batch(batch_inputs)
        
        # Store results
        predictions.extend(batch_predictions)
        inputs.extend(batch_inputs)
        
        # Store targets if available
        if 'target' in batch:
            if isinstance(batch['target'], list):
                targets.extend(batch['target'])
            else:
                targets.append(batch['target'])
        
        # Periodic cache clearing
        if (i // batch_size) % 10 == 0:
            clear_gpu_cache()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Generated {len(predictions)} predictions in {total_time:.2f}s")
    print(f"Average speed: {len(predictions)/total_time:.2f} samples/second")
    
    # Prepare output dataset
    output_data = {
        'input': inputs,
        'pred': predictions,
    }
    
    if targets:
        output_data['target'] = targets
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save as HuggingFace dataset
    from datasets import Dataset
    output_dataset = Dataset.from_dict(output_data)
    output_dataset.save_to_disk(args.output_dir)
    
    # Save metadata
    metadata = {
        'model_path': args.model_path,
        'dataset_name': args.dataset_name,
        'split': args.split,
        'num_samples': len(predictions),
        'batch_size': batch_size,
        'generation_time': total_time,
        'samples_per_second': len(predictions) / total_time,
        'args': vars(args)
    }
    
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {args.output_dir}")
    print("Sample predictions:")
    for i in range(min(3, len(predictions))):
        print(f"Input: {inputs[i]}")
        print(f"Prediction: {predictions[i]}")
        if targets and i < len(targets):
            print(f"Target: {targets[i]}")
        print("-" * 50)
    
    # Log to wandb
    wandb.log({
        "num_samples": len(predictions),
        "generation_time": total_time,
        "samples_per_second": len(predictions) / total_time,
        "batch_size": batch_size,
    })
    
    wandb.finish()


if __name__ == "__main__":
    main()
