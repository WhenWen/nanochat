#!/usr/bin/env bash

set -euo pipefail

export OMP_NUM_THREADS=1
: "${NANOCHAT_BASE_DIR:?Set NANOCHAT_BASE_DIR to the experiment artifact directory}"

if (( $# == 0 )); then
    DEPTHS=(12 16)
else
    DEPTHS=("$@")
fi
RESULTS_DIR="${NANOCHAT_BASE_DIR}/fixed_yoco_ablation"
mkdir -p "${RESULTS_DIR}"

run_model() {
    local depth="$1"
    local variant="$2"
    local device_batch_size=32
    local fixed_yoco_arg=()

    if (( depth >= 20 )); then
        device_batch_size=16
    fi
    if [[ "${variant}" == "fixed_yoco" ]]; then
        fixed_yoco_arg=(--fixed-yoco)
    fi

    local model_tag="fixed-yoco-d${depth}-${variant}"
    .venv/bin/torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
        --depth="${depth}" \
        --run=dummy \
        --model-tag="${model_tag}" \
        --device-batch-size="${device_batch_size}" \
        --core-metric-every=999999 \
        --core-metric-max-per-task=-1 \
        --sample-every=-1 \
        --save-every=-1 \
        "${fixed_yoco_arg[@]}" \
        2>&1 | tee "${RESULTS_DIR}/${model_tag}.log"
}

for depth in "${DEPTHS[@]}"; do
    run_model "${depth}" baseline
    run_model "${depth}" fixed_yoco
done
