#!/usr/bin/env bash

# Compare fixed-YOCO with the strongest README pretraining baseline (a825e63).
# The caller must provide an already prepared NANOCHAT_BASE_DIR.
set -euo pipefail

export OMP_NUM_THREADS=1
: "${NANOCHAT_BASE_DIR:?Set NANOCHAT_BASE_DIR to a prepared nanochat cache}"

common_train_args=(
    --depth=24
    --target-param-data-ratio=8
    --device-batch-size=16
    --fp8
    --run=dummy
)

baseline_tag=readme-a825e63-d24-r8-fp8-baseline
fixed_yoco_tag=readme-a825e63-d24-r8-fp8-fixed-yoco

torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    "${common_train_args[@]}" \
    --model-tag="$baseline_tag"
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- \
    --model-tag="$baseline_tag" \
    --device-batch-size=16

torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    "${common_train_args[@]}" \
    --model-tag="$fixed_yoco_tag" \
    --fixed-yoco
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- \
    --model-tag="$fixed_yoco_tag" \
    --device-batch-size=16
