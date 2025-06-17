#!/usr/bin/env bash

# run_eval.sh - Run evaluation script for Dolly-v2-3b model
# 
# Usage examples:
#   ./run_eval.sh --all                                    # Run all benchmarks on base model  
#   ./run_eval.sh --performance                            # Run performance benchmarks only
#   ./run_eval.sh --fairness                               # Run fairness benchmarks only
#   ./run_eval.sh --model_path ./dpo_output --hellaswag   # Run HellaSwag on fine-tuned model
#   ./run_eval.sh --arc_easy --boolq                      # Run specific benchmarks

set -e

# Check if virtual environment exists, create if not
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source .venv/Scripts/activate
else
    # Unix/Linux/MacOS
    source .venv/bin/activate
fi

# Upgrade pip and install requirements
echo "Installing/updating dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Run evaluation script with all provided arguments
echo "Starting evaluation..."
python evaluate.py "$@"

echo "Evaluation completed!" 