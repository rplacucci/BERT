#!/usr/bin/env bash
set -euo pipefail

# Default config
BERT_CONFIG="tiny"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bert_config)
            BERT_CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$BERT_CONFIG" ]]; then
    echo "Error: --bert_config argument is required."
    exit 1
fi

# List of GLUE tasks to evaluate
TASKS=(
  cola
  mnli-m
  mnli-mm
  mrpc
  qqp
  qnli
  rte
  sst2
  stsb
  wnli
  ax
)

# Loop over each task and invoke torchrun
for TASK in "${TASKS[@]}"; do
  echo "=========================================================="
  echo " Running GLUE tuning on $TASK for model BERT-$BERT_CONFIG"
  echo "=========================================================="
  torchrun --standalone --nproc-per-node=4 tune.py --task_name "$TASK" --bert_config "$BERT_CONFIG"
done

echo "Done tuning BERT-{$BERT_CONFIG}!"