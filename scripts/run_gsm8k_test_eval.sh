#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/eryx/verl/.venv/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_CACHE_ROOT=/home/eryx/verl-install/opd-preflight/vllm-cache
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
: "${EVAL_DIR:?Set a fresh eval directory}"
# guard appends a Ray temp override; this standalone vLLM evaluator does not use Ray.
for arg in "$@"; do
    [[ "$arg" == +ray_kwargs.ray_init._temp_dir=* ]] || { echo "Unexpected argument: $arg" >&2; exit 1; }
done
exec /home/eryx/verl/.venv/bin/python /home/eryx/qwen3.5-opd/scripts/eval_gsm8k_test.py --output "$EVAL_DIR"
