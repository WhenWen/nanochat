#!/usr/bin/env bash

set -euo pipefail

: "${NANOCHAT_BASE_DIR:?Set NANOCHAT_BASE_DIR to the experiment artifact directory}"

# Match nanochat's speedrun data recipe: bootstrap the tokenizer from 8 shards
# while the remaining pretraining shards download in parallel.
.venv/bin/python -m nanochat.dataset -n 8 -w 8
.venv/bin/python -m nanochat.dataset -n 170 -w 8 &
dataset_download_pid=$!
.venv/bin/python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768
wait "${dataset_download_pid}"
