# NanoYOCO: fixed-midpoint K/V reuse in nanochat

## TL;DR

Fixed-midpoint K/V reuse improved validation bits per byte (BPB) at nanochat depths 12, 16, 20, and 24. The relative gains were 0.108%, 0.137%, 0.107%, and 0.145%. Parameter counts were unchanged, and ordinary training time changed by less than 0.3% in every pair.

The downstream CORE result was mixed. CORE improved slightly at d12 and d16, then declined at d20 and d24. These are single-seed experiments, so the BPB result warrants replication and the architecture is not a demonstrated quality win yet.

This experiment is a parameter-preserving variant motivated by [YOCO](https://arxiv.org/abs/2405.05254), not a reproduction of the full YOCO decoder-decoder architecture. It changes the source of upper-layer keys and values while retaining nanochat's blocks, attention layers, and parameter count.

## Method

Let a model have `L` transformer blocks indexed from `0` to `L - 1`, and let `m = ceil(L / 2)`. The lower blocks run normally. After block `m - 1`, save its output as `z`.

For every upper block `l >= m`, queries use the current residual stream `h_l`, while keys and values use the same midpoint source `z`:

```text
q_l = W^Q_l Norm(h_l)
k_l = W^K_l Norm(z)
v_l = W^V_l Norm(z)
```

Each upper block keeps its own K and V matrices. The intervention therefore changes no parameter counts. The MLP, residual lambdas, x0 residual, sliding-window schedule, QK normalization, and ResFormer-style value embeddings are unchanged. The value-residual gate remains conditioned on the current query-side normalized state.

The shared source removes the sequential dependency between upper-layer K/V inputs. A specialized prefill path could project `z` through the stacked upper-layer K and V weights in two large GEMMs after the lower half completes. This repository does not implement that inference kernel, so the experiments below measure model quality and ordinary training cost, not prefill latency.

## Experimental setup

All eight runs used nanochat commit [`92d63d4e`](https://github.com/karpathy/nanochat/tree/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd), sequence length 2,048, the standard ClimbMix data recipe, and the default target of 12 training tokens per transformer scaling parameter. Each control/fixed pair used the same tokenizer, data order, seed, optimizer schedule, and final CORE evaluation.

The experiments ran on one node with eight H100 80GB GPUs. d12 and d16 used device batch 32. d20 and d24 used device batch 16. Total batch size and learning-rate scaling remained automatic. Each model was evaluated at one seed.

## Results

| Depth | Parameters | Validation BPB, control | Validation BPB, fixed | Relative BPB change | CORE, control | CORE, fixed | Training time, control | Training time, fixed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 286,261,730 | 0.846392 | 0.845479 | -0.108% | 0.1447 | 0.1468 | 5.38 min | 5.39 min |
| 16 | 536,871,738 | 0.780743 | 0.779677 | -0.137% | 0.2031 | 0.2036 | 20.90 min | 20.91 min |
| 20 | 896,533,746 | 0.736855 | 0.736065 | -0.107% | 0.2437 | 0.2391 | 68.48 min | 68.62 min |
| 24 | 1,384,122,122 | 0.702933 | 0.701914 | -0.145% | 0.2886 | 0.2727 | 176.59 min | 176.71 min |

The BPB direction replicated at all four depths, with a narrow relative improvement range of 0.107% to 0.145%. Fixed midpoint reuse was slightly worse early in some runs and finished better at every depth.

CORE improved by 0.0021 at d12 and 0.0005 at d16. It declined by 0.0046 at d20 and 0.0159 at d24. The larger-scale CORE regressions prevent a clean quality conclusion from these runs.

Peak memory increased by 79 to 128 MiB within each pair. Mean training throughput changed by -0.139% at d12 and -0.043% at d16; the d20 and d24 wall-clock deltas were +0.14 and +0.12 minutes. These measurements show no meaningful training-speed benefit, as expected from a training graph that still executes every projection and block normally.

## Reproduction

Install nanochat's GPU and development dependencies, then prepare the data and tokenizer:

```bash
uv sync --extra gpu --group dev
export NANOCHAT_BASE_DIR=/path/to/nanoyoco-artifacts
bash runs/prepare_fixed_yoco_ablation.sh
```

Run any subset of the tested depths on eight GPUs. With no arguments, the script runs d12 and d16.

```bash
bash runs/fixed_yoco_ablation.sh 12 16 20 24
```

The script runs each control immediately before its fixed-YOCO counterpart and writes logs under `${NANOCHAT_BASE_DIR}/fixed_yoco_ablation`. It uses final-only checkpoints and full final CORE evaluation. d20 and d24 automatically reduce the per-device batch from 32 to 16.

The wiring regression test runs without FlashAttention-3:

```bash
uv run pytest tests/test_fixed_yoco.py
```

## Limitations and next experiments

- Each comparison uses one seed. The BPB effect is small enough that seed replication is required.
- CORE does not improve consistently. d20 and d24 are negative downstream results.
- No optimized prefill implementation or latency benchmark exists in this fork. The intended inference benefit remains a structural hypothesis.
- Full YOCO uses a self-decoder and a cross-decoder with a global K/V cache. NanoYOCO preserves nanochat's original decoder and only reuses the midpoint residual as the input to upper-layer K/V projections.

The next useful experiment is a multi-seed d16 or d24 replication. If the BPB gain survives, an inference-only path can stack upper-layer K/V weights, materialize their caches after the midpoint, and measure prefill latency across context lengths.

## References

- Sun et al., [YOCO: You Only Cache Once](https://arxiv.org/abs/2405.05254)
- [nanochat](https://github.com/karpathy/nanochat)
- [Marin fixed-YOCO experiment](https://github.com/marin-community/marin/issues/8196)
