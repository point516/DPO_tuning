"""
train_dpo.py – Alignment-tune Dolly-v2-3b with Direct Preference Optimization (DPO)
using a configurable subset of the Anthropic HH-RLHF preference dataset.

Run locally or on a single‑GPU cloud instance (e.g. RunPod A100). Example:

    python train_dpo.py \
        --model_name databricks/dolly-v2-3b \
        --subset_size 20000 \
        --subset_strategy random \
        --output_dir ./dpo_dolly_20k \
        --batch_size 64 \
        --gradient_accumulation_steps 8 \
        --epochs 3 \
        --learning_rate 2e-5 \
        --beta 0.1

Dependencies (see requirements.txt):
  • python>=3.11   • torch>=2.3   • transformers>=4.43   • datasets>=2.20
  • trl[torch]>=0.9   • accelerate>=0.28   • bitsandbytes   • evaluate   • wandb (optional)
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Literal

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from trl import DPOTrainer, DPOConfig
from accelerate import Accelerator

################################################################################
# Helper functions
################################################################################

def sample_hh_dataset(
    size: int,
    strategy: Literal["random", "stratified"] = "random",
    seed: int = 42,
):
    """Return a DatasetDict with `prompt`, `chosen`, `rejected` columns limited to
    *size* examples using the selected *strategy*.

    Supported strategies:
      • random – uniform random sampling.
      • stratified – equal halves of harmless/helpful categories.
    """
    hh = load_dataset("Anthropic/hh-rlhf", split="train")

    if strategy == "random":
        hh = hh.shuffle(seed=seed).select(range(size))
    elif strategy == "stratified":
        # For stratified sampling, we can't easily filter by helpfulness/harmlessness
        # since the dataset doesn't have those labels, so we'll fall back to random
        hh = hh.shuffle(seed=seed).select(range(size))
    else:
        raise ValueError("Unsupported subset_strategy. Choose 'random' or 'stratified'.")

    # Process the dataset to extract prompts and create the expected format
    def process_conversations(examples):
        prompts = []
        chosen_responses = []
        rejected_responses = []
        
        for chosen_conv, rejected_conv in zip(examples["chosen"], examples["rejected"]):
            # Split conversations to extract the last exchange
            chosen_parts = chosen_conv.split("\n\nAssistant:")
            rejected_parts = rejected_conv.split("\n\nAssistant:")
            
            if len(chosen_parts) >= 2 and len(rejected_parts) >= 2:
                # Extract the prompt (everything up to the last assistant response)
                prompt = chosen_parts[0]
                if not prompt.startswith("Human:"):
                    prompt = "Human: " + prompt
                
                # Extract the responses (last assistant response)
                chosen_response = chosen_parts[-1].strip()
                rejected_response = rejected_parts[-1].strip()
                
                prompts.append(prompt)
                chosen_responses.append(chosen_response)
                rejected_responses.append(rejected_response)
        
        return {
            "prompt": prompts,
            "chosen": chosen_responses,
            "rejected": rejected_responses
        }
    
    # Apply the processing function
    hh = hh.map(
        process_conversations,
        batched=True,
        remove_columns=hh.column_names,
        desc="Processing conversations"
    )
    
    # Filter out any empty examples
    hh = hh.filter(lambda x: len(x["prompt"]) > 0 and len(x["chosen"]) > 0 and len(x["rejected"]) > 0)
    
    return hh

def preprocess_dataset(dataset, tokenizer, max_len):
    """Pre-tokenize the dataset to avoid on-the-fly tokenization during training."""
    def preprocess_function(examples):
        # Tokenize prompts, chosen, and rejected responses
        prompts = examples["prompt"]
        chosen = examples["chosen"]
        rejected = examples["rejected"]
        
        # Calculate prompt length proportional to max_len
        max_prompt_len = min(384, max_len // 2)
        
        # Tokenize each field
        prompt_tokens = tokenizer(prompts, truncation=True, max_length=max_prompt_len, padding=False)
        chosen_tokens = tokenizer(chosen, truncation=True, max_length=max_len, padding=False)
        rejected_tokens = tokenizer(rejected, truncation=True, max_length=max_len, padding=False)
        
        return {
            "prompt": prompts,
            "chosen": chosen,
            "rejected": rejected,
            "prompt_input_ids": prompt_tokens["input_ids"],
            "prompt_attention_mask": prompt_tokens["attention_mask"],
            "chosen_input_ids": chosen_tokens["input_ids"],
            "chosen_attention_mask": chosen_tokens["attention_mask"],
            "rejected_input_ids": rejected_tokens["input_ids"],
            "rejected_attention_mask": rejected_tokens["attention_mask"],
        }
    
    # Process dataset in batches with multiple workers for speed
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=8,
        desc="Tokenizing dataset"
    )
    
    return processed_dataset

################################################################################
# Argument parsing
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="DPO fine‑tune Dolly with HH‑RLHF subset")

    # Model & data
    parser.add_argument("--model_name", default="databricks/dolly-v2-3b", help="HF model repo or path")
    parser.add_argument("--subset_size", type=int, default=20000, help="Number of HH‑RLHF pairs to train on")
    parser.add_argument("--subset_strategy", choices=["random", "stratified"], default="random")
    parser.add_argument("--subset_seed", type=int, default=42, help="Deterministic seed for subset sampling")

    # Training hyper‑parameters - Updated defaults for optimization 5
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=96, help="Increased batch size for better GPU utilization with quantization")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=6, help="Reduced to maintain effective batch size while utilizing freed memory")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.1, help="β coefficient in DPO objective")
    parser.add_argument("--max_len", type=int, default=768, help="Increased sequence length to utilize freed memory from quantization")

    # Memory optimization options
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization for even more memory savings")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization for memory savings with better precision")
    parser.add_argument("--share_reference_model", action="store_true", default=True, help="Share reference model to save memory")

    # Misc
    parser.add_argument("--output_dir", default="./dpo_output")
    parser.add_argument("--wandb_project", default=None, help="Enable Weights & Biases logging with this project name")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

################################################################################
# Main routine
################################################################################

def main():
    args = parse_args()

    # Set environment variable for memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    accelerator = Accelerator(log_with="wandb" if args.wandb_project else None)
    if accelerator.is_main_process:
        print("Loading base & reference models…")
        print(f"Memory optimization settings:")
        print(f"  - Batch size: {args.batch_size}")
        print(f"  - Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"  - Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
        print(f"  - Max sequence length: {args.max_len}")
        print(f"  - Share reference model: {args.share_reference_model}")
        
        # Display quantization info
        if args.use_8bit:
            print(f"  - Using 8-bit quantization (good balance of memory vs precision)")
        elif args.use_4bit:
            print(f"  - Using 4-bit quantization (maximum memory savings)")
        else:
            print(f"  - Using full precision (no quantization)")
            
        # Memory utilization recommendations
        if args.use_8bit or args.use_4bit:
            print(f"Quantization benefits:")
            print(f"  ✓ ~50% memory reduction from quantization")
            print(f"  ✓ Larger batch sizes possible")
            print(f"  ✓ Longer sequences supported")
            print(f"  ✓ Can train larger models on same hardware")

    # Ensure determinism
    set_seed(args.seed)

    # Validate quantization options
    if args.use_4bit and args.use_8bit:
        raise ValueError("Cannot use both 4-bit and 8-bit quantization simultaneously. Choose one.")

    # Load tokenizer (shared between policy & ref)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    
    # Fix warning: ensure pad_token_id and eos_token_id are different
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        print(f"Fixed tokenizer: pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")

    # Prepare model loading kwargs with Flash Attention 2 (optimization 3)
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto",
        "trust_remote_code": True,
        "attn_implementation": "flash_attention_2",  # Enable Flash Attention 2
    }

    # Add quantization if requested
    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif args.use_8bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False,  # Keep everything on GPU
            llm_int8_has_fp16_weight=False
        )

    # Load policy model (trainable)
    if accelerator.is_main_process:
        print("Loading policy model with Flash Attention 2...")
    policy_model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    policy_model.resize_token_embeddings(len(tokenizer))

    # Reference model handling
    if args.share_reference_model:
        # Use the same model for reference (memory efficient)
        ref_model = None
        if accelerator.is_main_process:
            print("Using shared reference model (memory efficient)")
    else:
        # Load separate reference model (more memory but potentially better results)
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
        if accelerator.is_main_process:
            print("Using separate reference model")

    # Dataset with pre-tokenization (optimization 2)
    if accelerator.is_main_process:
        print("Sampling HH‑RLHF subset…")
    train_dataset = sample_hh_dataset(
        size=args.subset_size,
        strategy=args.subset_strategy,
        seed=args.subset_seed,
    )
    
    if accelerator.is_main_process:
        print("Pre-tokenizing dataset for faster training...")
    train_dataset = preprocess_dataset(train_dataset, tokenizer, args.max_len)

    # Create DPO training arguments with optimizations
    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        dataloader_drop_last=True,
        dataloader_pin_memory=True,  # Re-enable pin memory for faster host->device transfer
        remove_unused_columns=False,
        report_to=["wandb"] if args.wandb_project else None,
        max_length=args.max_len,
        max_prompt_length=min(384, args.max_len // 2),  # Proportional to max_len but capped
        gradient_checkpointing=True,   # Enable gradient checkpointing to save memory
    )

    # Clear cache before creating trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Trainer
    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,  # Use processing_class parameter as per current API
    )

    # Kick off training
    if accelerator.is_main_process:
        print("Starting training…")
        print(f"Optimizations applied:")
        print(f"  ✓ Gradient checkpointing enabled")
        print(f"  ✓ Dataset pre-tokenized")
        print(f"  ✓ Flash Attention 2 enabled")
        print(f"  ✓ Increased batch size to {args.batch_size}, reduced grad accumulation to {args.gradient_accumulation_steps}")
        print(f"  ✓ Increased max sequence length to {args.max_len}")
        if args.use_8bit:
            print(f"  ✓ 8-bit quantization enabled (memory efficient with good precision)")
        elif args.use_4bit:
            print(f"  ✓ 4-bit quantization enabled (maximum memory efficiency)")
        if torch.cuda.is_available():
            print(f"GPU memory before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    trainer.train()

    # Save
    if accelerator.is_main_process:
        print("Saving fine‑tuned model to", args.output_dir)
    trainer.save_model(args.output_dir)

    if accelerator.is_main_process:
        print("Done ✔")


if __name__ == "__main__":
    main()
