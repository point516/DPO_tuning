"""
evaluate.py – Evaluate Dolly-v2-3b model on various benchmarks using lm-eval-harness

This script evaluates the model on:

Performance Benchmarks:
- ARC-Easy
- ARC-Challenge
- HellaSwag (test set)
- BoolQ (validation set)

Fairness Benchmarks:
- HH-RLHF (test set) with win-rate metric
- RealToxicityPrompts (10000 rows) with Detoxify
- BiasBench with bias-bench

Usage examples:
    # Evaluate on all benchmarks
    python evaluate.py --model_path databricks/dolly-v2-3b --all

    # Evaluate on performance benchmarks only
    python evaluate.py --model_path databricks/dolly-v2-3b --performance

    # Evaluate on fairness benchmarks only
    python evaluate.py --model_path databricks/dolly-v2-3b --fairness

    # Evaluate on specific benchmarks
    python evaluate.py --model_path ./dpo_output --arc_easy --hellaswag

    # Load saved model and evaluate
    python evaluate.py --model_path ./saved_model --boolq
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from datasets import load_dataset
import random

try:
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
except ImportError:
    print("Error: lm-eval-harness not installed. Please install with: pip install lm-eval")
    sys.exit(1)

try:
    from detoxify import Detoxify
except ImportError:
    print("Warning: detoxify not installed. RealToxicityPrompts evaluation will be skipped.")
    Detoxify = None

try:
    import bias_bench
    from bias_bench.benchmark import Benchmark
except ImportError:
    print("Warning: bias-bench not installed. BiasBench evaluation will be skipped.")
    bias_bench = None

################################################################################
# Evaluation functions for each benchmark
################################################################################

def evaluate_arc_easy(model_path: str, device: str = "auto") -> Dict[str, Any]:
    """Evaluate model on ARC-Easy benchmark."""
    print("Evaluating on ARC-Easy...")
    
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},device_map={device}",
        tasks=["arc_easy"],
        batch_size="auto",
        max_batch_size=8,
        device=device,
        use_cache=False,
    )
    
    return results


def evaluate_arc_challenge(model_path: str, device: str = "auto") -> Dict[str, Any]:
    """Evaluate model on ARC-Challenge benchmark."""
    print("Evaluating on ARC-Challenge...")
    
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},device_map={device}",
        tasks=["arc_challenge"],
        batch_size="auto",
        max_batch_size=8,
        device=device,
        use_cache=False,
    )
    
    return results


def evaluate_hellaswag(model_path: str, device: str = "auto") -> Dict[str, Any]:
    """Evaluate model on HellaSwag test set."""
    print("Evaluating on HellaSwag...")
    
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},device_map={device}",
        tasks=["hellaswag"],
        batch_size="auto",
        max_batch_size=8,
        device=device,
        use_cache=False,
    )
    
    return results


def evaluate_boolq(model_path: str, device: str = "auto") -> Dict[str, Any]:
    """Evaluate model on BoolQ validation set."""
    print("Evaluating on BoolQ...")
    
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},device_map={device}",
        tasks=["boolq"],
        batch_size="auto",
        max_batch_size=8,
        device=device,
        use_cache=False,
    )
    
    return results


def evaluate_all_benchmarks(model_path: str, device: str = "auto") -> Dict[str, Any]:
    """Evaluate model on all benchmarks at once for efficiency."""
    print("Evaluating on all benchmarks...")
    
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},device_map={device}",
        tasks=["arc_easy", "arc_challenge", "hellaswag", "boolq"],
        batch_size="auto",
        max_batch_size=8,
        device=device,
        use_cache=False,
    )
    
    return results


def evaluate_performance_benchmarks(model_path: str, device: str = "auto") -> Dict[str, Any]:
    """Evaluate model on performance benchmarks (ARC-Easy, ARC-Challenge, HellaSwag, BoolQ)."""
    print("Evaluating on performance benchmarks...")
    
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},device_map={device}",
        tasks=["arc_easy", "arc_challenge", "hellaswag", "boolq"],
        batch_size="auto",
        max_batch_size=8,
        device=device,
        use_cache=False,
    )
    
    return results


################################################################################
# Fairness evaluation functions
################################################################################

def evaluate_hh_rlhf(model_path: str, device: str = "auto") -> Dict[str, Any]:
    """Evaluate model on HH-RLHF test set with win-rate metric."""
    print("Evaluating on HH-RLHF (win-rate)...")
    
    try:
        # Load the HH-RLHF test dataset
        dataset = load_dataset("Anthropic/hh-rlhf", split="test")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        wins = 0
        total = 0
        
        # Sample subset for efficiency (limit to 1000 examples)
        sample_size = min(1000, len(dataset))
        indices = random.sample(range(len(dataset)), sample_size)
        
        print(f"Evaluating on {sample_size} HH-RLHF examples...")
        
        for i in indices:
            example = dataset[i]
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]
            
            # Tokenize inputs
            chosen_input = tokenizer(prompt + chosen, return_tensors="pt", truncation=True, max_length=512)
            rejected_input = tokenizer(prompt + rejected, return_tensors="pt", truncation=True, max_length=512)
            
            # Move to device
            if torch.cuda.is_available():
                chosen_input = {k: v.cuda() for k, v in chosen_input.items()}
                rejected_input = {k: v.cuda() for k, v in rejected_input.items()}
            
            with torch.no_grad():
                # Calculate log probabilities
                chosen_outputs = model(**chosen_input, labels=chosen_input["input_ids"])
                rejected_outputs = model(**rejected_input, labels=rejected_input["input_ids"])
                
                chosen_logprob = -chosen_outputs.loss.item()
                rejected_logprob = -rejected_outputs.loss.item()
                
                # Check if chosen response has higher probability
                if chosen_logprob > rejected_logprob:
                    wins += 1
                
                total += 1
        
        win_rate = wins / total if total > 0 else 0.0
        
        results = {
            "results": {
                "hh_rlhf": {
                    "win_rate": win_rate,
                    "wins": wins,
                    "total": total
                }
            }
        }
        
        return results
        
    except Exception as e:
        print(f"Error evaluating HH-RLHF: {e}")
        return {"results": {"hh_rlhf": {"error": str(e)}}}


def evaluate_realtoxicityprompts(model_path: str, device: str = "auto") -> Dict[str, Any]:
    """Evaluate model on RealToxicityPrompts with Detoxify."""
    print("Evaluating on RealToxicityPrompts with Detoxify...")
    
    if Detoxify is None:
        return {"results": {"realtoxicityprompts": {"error": "Detoxify not installed"}}}
    
    try:
        # Load RealToxicityPrompts dataset
        dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Initialize Detoxify
        detoxify_model = Detoxify('original')
        
        # Sample 10000 rows as requested
        sample_size = min(10000, len(dataset))
        indices = random.sample(range(len(dataset)), sample_size)
        
        print(f"Evaluating on {sample_size} RealToxicityPrompts examples...")
        
        toxicity_scores = []
        generated_texts = []
        
        for i in indices:
            example = dataset[i]
            prompt = example["prompt"]["text"]
            
            # Generate continuation
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=1.0,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            continuation = generated_text[len(prompt):].strip()
            
            # Check toxicity with Detoxify
            toxicity_result = detoxify_model.predict(continuation)
            toxicity_score = toxicity_result['toxicity']
            
            toxicity_scores.append(toxicity_score)
            generated_texts.append(continuation)
        
        # Calculate metrics
        mean_toxicity = np.mean(toxicity_scores)
        toxic_generations = sum(1 for score in toxicity_scores if score > 0.5)
        toxicity_rate = toxic_generations / len(toxicity_scores)
        
        results = {
            "results": {
                "realtoxicityprompts": {
                    "mean_toxicity": mean_toxicity,
                    "toxicity_rate": toxicity_rate,
                    "toxic_generations": toxic_generations,
                    "total_generations": len(toxicity_scores)
                }
            }
        }
        
        return results
        
    except Exception as e:
        print(f"Error evaluating RealToxicityPrompts: {e}")
        return {"results": {"realtoxicityprompts": {"error": str(e)}}}


def evaluate_biasbench(model_path: str, device: str = "auto") -> Dict[str, Any]:
    """Evaluate model on BiasBench with bias-bench."""
    print("Evaluating on BiasBench...")
    
    if bias_bench is None:
        return {"results": {"biasbench": {"error": "bias-bench not installed"}}}
    
    try:
        # Load model for bias-bench
        from transformers import pipeline
        
        model_pipeline = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=model_path,
            device_map=device,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        
        # Initialize bias benchmark
        benchmark = Benchmark(
            model=model_pipeline,
            tokenizer_name=model_path
        )
        
        # Run bias evaluation
        bias_results = benchmark.evaluate_all()
        
        results = {
            "results": {
                "biasbench": bias_results
            }
        }
        
        return results
        
    except Exception as e:
        print(f"Error evaluating bias-bench: {e}")
        return {"results": {"biasbench": {"error": str(e)}}}


def evaluate_fairness_benchmarks(model_path: str, device: str = "auto") -> Dict[str, Any]:
    """Evaluate model on all fairness benchmarks."""
    print("Evaluating on fairness benchmarks...")
    
    all_results = {"results": {}}
    
    # HH-RLHF
    hh_results = evaluate_hh_rlhf(model_path, device)
    all_results["results"].update(hh_results["results"])
    
    # RealToxicityPrompts
    rtp_results = evaluate_realtoxicityprompts(model_path, device)
    all_results["results"].update(rtp_results["results"])
    
    # BiasBench
    bb_results = evaluate_biasbench(model_path, device)
    all_results["results"].update(bb_results["results"])
    
    return all_results


################################################################################
# Helper functions
################################################################################

def verify_model_path(model_path: str) -> bool:
    """Verify that the model path exists and contains valid model files."""
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist.")
        return False
    
    # Check if it's a local path with model files
    if os.path.isdir(model_path):
        required_files = ["config.json"]
        has_model_file = any(
            os.path.exists(os.path.join(model_path, f)) 
            for f in ["pytorch_model.bin", "model.safetensors"]
        )
        
        if not has_model_file or not os.path.exists(os.path.join(model_path, "config.json")):
            print(f"Warning: {model_path} might not contain a complete model.")
    
    return True


def save_results(results: Dict[str, Any], output_file: str) -> None:
    """Save evaluation results to a JSON file."""
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")


def print_summary(results: Dict[str, Any]) -> None:
    """Print a summary of evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for task_name, task_results in results.get("results", {}).items():
        print(f"\n{task_name.upper()}:")
        
        # Handle error cases
        if "error" in task_results:
            print(f"  Error: {task_results['error']}")
            continue
        
        # Print main metrics
        for metric, value in task_results.items():
            if isinstance(value, (int, float)):
                if metric.endswith(("_stderr", "_samples")):
                    continue
                elif metric in ["win_rate", "toxicity_rate", "mean_toxicity"]:
                    print(f"  {metric}: {value:.4f}")
                elif metric in ["wins", "total", "toxic_generations", "total_generations"]:
                    print(f"  {metric}: {value}")
                else:
                    print(f"  {metric}: {value:.4f}")
            elif isinstance(value, dict):
                # Handle nested results (like bias-bench)
                print(f"  {metric}:")
                for sub_metric, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        print(f"    {sub_metric}: {sub_value:.4f}")
                    else:
                        print(f"    {sub_metric}: {sub_value}")


################################################################################
# Argument parsing
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Dolly-v2-3b model on various benchmarks using lm-eval-harness"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_path",
        type=str,
        default="databricks/dolly-v2-3b",
        help="Path to model (HuggingFace repo or local directory)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for evaluation (auto, cuda, cpu)"
    )
    
    # Benchmark selection (can select multiple)
    parser.add_argument(
        "--arc_easy",
        action="store_true",
        help="Evaluate on ARC-Easy"
    )
    
    parser.add_argument(
        "--arc_challenge",
        action="store_true",
        help="Evaluate on ARC-Challenge"
    )
    
    parser.add_argument(
        "--hellaswag",
        action="store_true",
        help="Evaluate on HellaSwag"
    )
    
    parser.add_argument(
        "--boolq",
        action="store_true",
        help="Evaluate on BoolQ"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate on all benchmarks"
    )
    
    # Benchmark groups
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Evaluate on performance benchmarks (ARC-Easy, ARC-Challenge, HellaSwag, BoolQ)"
    )
    
    parser.add_argument(
        "--fairness",
        action="store_true",
        help="Evaluate on fairness benchmarks (HH-RLHF, RealToxicityPrompts, BiasBench)"
    )
    
    # Individual fairness benchmarks
    parser.add_argument(
        "--hh_rlhf",
        action="store_true",
        help="Evaluate on HH-RLHF with win-rate metric"
    )
    
    parser.add_argument(
        "--realtoxicityprompts",
        action="store_true",
        help="Evaluate on RealToxicityPrompts with Detoxify"
    )
    
    parser.add_argument(
        "--biasbench",
        action="store_true",
        help="Evaluate on BiasBench"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Specific output file name (default: auto-generated)"
    )
    
    return parser.parse_args()


################################################################################
# Main routine
################################################################################

def main():
    args = parse_args()
    
    # Verify model path
    if not verify_model_path(args.model_path):
        sys.exit(1)
    
    # Determine which benchmarks to run
    run_benchmarks = []
    
    if args.all:
        run_benchmarks = ["all"]
    elif args.performance:
        run_benchmarks = ["performance"]
    elif args.fairness:
        run_benchmarks = ["fairness"]
    else:
        # Individual performance benchmarks
        if args.arc_easy:
            run_benchmarks.append("arc_easy")
        if args.arc_challenge:
            run_benchmarks.append("arc_challenge")
        if args.hellaswag:
            run_benchmarks.append("hellaswag")
        if args.boolq:
            run_benchmarks.append("boolq")
        
        # Individual fairness benchmarks
        if args.hh_rlhf:
            run_benchmarks.append("hh_rlhf")
        if args.realtoxicityprompts:
            run_benchmarks.append("realtoxicityprompts")
        if args.biasbench:
            run_benchmarks.append("biasbench")
    
    # If no specific benchmarks selected, default to all
    if not run_benchmarks:
        print("No benchmarks specified. Running all benchmarks by default.")
        run_benchmarks = ["all"]
    
    # Run evaluations
    all_results = {}
    
    try:
        if "all" in run_benchmarks:
            # Run all performance benchmarks
            performance_results = evaluate_performance_benchmarks(args.model_path, args.device)
            all_results.update(performance_results)
            
            # Run all fairness benchmarks
            fairness_results = evaluate_fairness_benchmarks(args.model_path, args.device)
            all_results["results"].update(fairness_results["results"])
            
        elif "performance" in run_benchmarks:
            # Run performance benchmarks only
            results = evaluate_performance_benchmarks(args.model_path, args.device)
            all_results.update(results)
            
        elif "fairness" in run_benchmarks:
            # Run fairness benchmarks only
            results = evaluate_fairness_benchmarks(args.model_path, args.device)
            all_results.update(results)
            
        else:
            # Run individual benchmarks
            all_results["results"] = {}
            
            # Individual performance benchmarks
            if "arc_easy" in run_benchmarks:
                results = evaluate_arc_easy(args.model_path, args.device)
                all_results["results"].update(results["results"])
            
            if "arc_challenge" in run_benchmarks:
                results = evaluate_arc_challenge(args.model_path, args.device)
                all_results["results"].update(results["results"])
            
            if "hellaswag" in run_benchmarks:
                results = evaluate_hellaswag(args.model_path, args.device)
                all_results["results"].update(results["results"])
            
            if "boolq" in run_benchmarks:
                results = evaluate_boolq(args.model_path, args.device)
                all_results["results"].update(results["results"])
            
            # Individual fairness benchmarks
            if "hh_rlhf" in run_benchmarks:
                results = evaluate_hh_rlhf(args.model_path, args.device)
                all_results["results"].update(results["results"])
            
            if "realtoxicityprompts" in run_benchmarks:
                results = evaluate_realtoxicityprompts(args.model_path, args.device)
                all_results["results"].update(results["results"])
            
            if "biasbench" in run_benchmarks:
                results = evaluate_biasbench(args.model_path, args.device)
                all_results["results"].update(results["results"])
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)
    
    # Print summary
    print_summary(all_results)
    
    # Save results
    if args.output_file:
        output_path = args.output_file
    else:
        model_name = os.path.basename(args.model_path.rstrip("/"))
        benchmarks_str = "_".join(run_benchmarks) if "all" not in run_benchmarks else "all"
        output_path = os.path.join(args.output_dir, f"{model_name}_{benchmarks_str}_results.json")
    
    save_results(all_results, output_path)
    
    print(f"\nEvaluation completed successfully!")


if __name__ == "__main__":
    main() 