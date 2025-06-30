"""
calculate_logprobs.py – Calculate log-probabilities on HH-RLHF dataset samples
to determine whether a HuggingFace model prefers 'chosen' or 'rejected' responses.

This script:
1. Loads a HuggingFace language model
2. Takes 50 non-random samples from the HH-RLHF dataset
3. Calculates log-probabilities for both chosen and rejected responses
4. Reports which response the model assigns higher probability to

Usage:
    python calculate_logprobs.py --model_name databricks/dolly-v2-3b
    python calculate_logprobs.py --model_path ./dpo_output --num_samples 100
"""

from __future__ import annotations

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F


def load_model_and_tokenizer(model_path: str, device: str = "auto") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer from HuggingFace hub or local path."""
    print(f"Loading model and tokenizer from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    return model, tokenizer


def process_hh_conversations(examples: Dict) -> Dict[str, List[str]]:
    """Process HH-RLHF conversation format to extract prompts and responses."""
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


def calculate_log_probability(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                            prompt: str, response: str, device: str) -> float:
    """Calculate log-probability of a response given a prompt."""
    
    # Combine prompt and response
    full_text = prompt + "\n\nAssistant: " + response
    
    # Tokenize the full text
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(model.device)
    
    # Tokenize just the prompt to find where response starts
    prompt_text = prompt + "\n\nAssistant:"
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024)
    prompt_length = prompt_inputs["input_ids"].shape[1]
    
    # Calculate log probabilities
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        
        # Get log probabilities for the response tokens only
        # Shift logits and input_ids for next token prediction
        shift_logits = logits[0, prompt_length-1:-1, :]  # Start from prompt_length-1 to predict first response token
        shift_labels = input_ids[0, prompt_length:]
        
        # Calculate log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Get log probability of the actual tokens
        token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # Return average log probability
        avg_log_prob = token_log_probs.mean().item()
        
    return avg_log_prob


def evaluate_preferences(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                       dataset: Dict, num_samples: int = 50) -> Dict[str, any]:
    """Evaluate model preferences on HH-RLHF samples."""
    
    device = next(model.parameters()).device
    results = {
        "total_samples": 0,
        "chosen_preferred": 0,
        "rejected_preferred": 0,
        "equal_preference": 0,
        "samples": []
    }
    
    print(f"Evaluating preferences on {num_samples} samples...")
    
    for i in range(min(num_samples, len(dataset["prompt"]))):
        prompt = dataset["prompt"][i]
        chosen = dataset["chosen"][i]
        rejected = dataset["rejected"][i]
        
        print(f"Processing sample {i+1}/{num_samples}...")
        
        try:
            # Calculate log probabilities
            chosen_logprob = calculate_log_probability(model, tokenizer, prompt, chosen, device)
            rejected_logprob = calculate_log_probability(model, tokenizer, prompt, rejected, device)
            
            # Determine preference
            if chosen_logprob > rejected_logprob:
                preference = "chosen"
                results["chosen_preferred"] += 1
            elif rejected_logprob > chosen_logprob:
                preference = "rejected"
                results["rejected_preferred"] += 1
            else:
                preference = "equal"
                results["equal_preference"] += 1
            
            # Store sample results
            sample_result = {
                "sample_id": i,
                "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                "chosen_logprob": chosen_logprob,
                "rejected_logprob": rejected_logprob,
                "preference": preference,
                "logprob_diff": chosen_logprob - rejected_logprob
            }
            results["samples"].append(sample_result)
            
            print(f"  Chosen log-prob: {chosen_logprob:.4f}")
            print(f"  Rejected log-prob: {rejected_logprob:.4f}")
            print(f"  Model prefers: {preference}")
            print()
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    results["total_samples"] = len(results["samples"])
    
    return results


def print_summary(results: Dict[str, any]):
    """Print summary of evaluation results."""
    total = results["total_samples"]
    chosen_pref = results["chosen_preferred"]
    rejected_pref = results["rejected_preferred"]
    equal_pref = results["equal_preference"]
    
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total samples evaluated: {total}")
    print(f"Model prefers 'chosen' response: {chosen_pref} ({chosen_pref/total*100:.1f}%)")
    print(f"Model prefers 'rejected' response: {rejected_pref} ({rejected_pref/total*100:.1f}%)")
    print(f"Equal preference: {equal_pref} ({equal_pref/total*100:.1f}%)")
    print()
    
    if chosen_pref > rejected_pref:
        print(f"✅ Model aligns with human preferences ({chosen_pref/total*100:.1f}% preference for chosen responses)")
    elif rejected_pref > chosen_pref:
        print(f"❌ Model anti-aligns with human preferences ({rejected_pref/total*100:.1f}% preference for rejected responses)")
    else:
        print("🤷 Model shows no clear preference alignment")
    
    # Show average log-probability differences
    if results["samples"]:
        logprob_diffs = [sample["logprob_diff"] for sample in results["samples"]]
        avg_diff = np.mean(logprob_diffs)
        print(f"\nAverage log-probability difference (chosen - rejected): {avg_diff:.4f}")


def save_results(results: Dict[str, any], output_file: str):
    """Save detailed results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {output_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calculate log-probabilities on HH-RLHF samples")
    
    parser.add_argument("--model_name", type=str, help="HuggingFace model name (e.g., databricks/dolly-v2-3b)")
    parser.add_argument("--model_path", type=str, help="Path to local model directory")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to evaluate (default: 50)")
    parser.add_argument("--output_file", type=str, default="logprob_results.json", help="Output file for detailed results")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Validate that either model_name or model_path is provided
    if not args.model_name and not args.model_path:
        parser.error("Either --model_name or --model_path must be provided")
    
    # Use model_name if both are provided
    if args.model_name:
        args.model_path = args.model_name
    
    return args


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("HH-RLHF Log-Probability Evaluation")
    print("=" * 60)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    
    # Load HH-RLHF dataset
    print("Loading HH-RLHF dataset...")
    raw_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    
    # Take first N samples (non-random for reproducibility)
    raw_dataset = raw_dataset.select(range(min(args.num_samples, len(raw_dataset))))
    
    # Process the conversations
    print("Processing conversations...")
    processed_data = process_hh_conversations({
        "chosen": raw_dataset["chosen"],
        "rejected": raw_dataset["rejected"]
    })
    
    # Filter out empty examples
    valid_indices = [
        i for i in range(len(processed_data["prompt"]))
        if len(processed_data["prompt"][i]) > 0 and 
           len(processed_data["chosen"][i]) > 0 and 
           len(processed_data["rejected"][i]) > 0
    ]
    
    filtered_data = {
        "prompt": [processed_data["prompt"][i] for i in valid_indices],
        "chosen": [processed_data["chosen"][i] for i in valid_indices],
        "rejected": [processed_data["rejected"][i] for i in valid_indices]
    }
    
    print(f"Found {len(filtered_data['prompt'])} valid conversation pairs")
    
    # Evaluate preferences
    results = evaluate_preferences(model, tokenizer, filtered_data, args.num_samples)
    
    # Print summary
    print_summary(results)
    
    # Save detailed results
    save_results(results, args.output_file)


if __name__ == "__main__":
    main() 