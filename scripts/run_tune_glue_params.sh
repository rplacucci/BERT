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

# List of learning rates to tune with
LRS=(
  3e-4
  1e-4
  5e-5
  3e-5
)

# List of batch sizes to tune with
BSS=(
  8
  16
  32
  64
  128
)

# List of GLUE tasks to tune
TASKS=(
  cola
  mnli
  mrpc
  qqp
  qnli
  rte
  sst2
  stsb
  wnli
)

# Loop over each task and invoke torchrun
for TASK in "${TASKS[@]}"; do
  for LR in "${LRS[@]}"; do
    for BS in "${BSS[@]}"; do
      echo "===================================================================================================="
      echo " Running GLUE tuning on $TASK for model BERT-$BERT_CONFIG with lr=$LR and batch_size=$BS"
      echo "===================================================================================================="
      torchrun --standalone --nproc-per-node=4 tune.py --task_name "$TASK" --bert_config "$BERT_CONFIG" --lr "$LR" --batch_size "$BS"
    done
  done
done

echo "Done tuning BERT-$BERT_CONFIG!"