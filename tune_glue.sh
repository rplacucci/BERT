#!/usr/bin/env bash
set -euo pipefail

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
  echo "====================================="
  echo " Running GLUE tuning on: $TASK"
  echo "====================================="
  torchrun --standalone --nproc-per-node=4 tune.py --task_name "$TASK"
done

# Once all tasks are done, zip up the submission folder
echo "Zipping submission directory…"
zip -r submission.zip submission
echo "Created submission.zip"