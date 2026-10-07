#!/usr/bin/env bash
# Isolated Transformers CQ path. Does not patch or relaunch the accepted verl baseline.
set -euo pipefail
PROJECT=/home/eryx/qwen3.5-opd
PYTHON=/home/eryx/verl/.venv/bin/python
GUARD=/home/eryx/verl/examples/on_policy_distillation_trainer/diagnostics/guard.py
GUARD_TIMEOUT=${CQ_GUARD_TIMEOUT_SECONDS:-1800}
if [[ ! "$GUARD_TIMEOUT" =~ ^[1-9][0-9]{0,4}$ ]]; then
  echo 'CQ_GUARD_TIMEOUT_SECONDS must be a positive integer below 100000.' >&2
  exit 1
fi
cd "$PROJECT"
if pgrep -f '^/home/eryx/verl/.venv/bin/python .*diagnostics/guard.py' >/dev/null; then
  echo 'Another guarded GPU job is active; refusing overlap.' >&2
  exit 1
fi
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$PROJECT/.cq-deps:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_CACHE_ROOT=/home/eryx/verl-install/opd-preflight/vllm-cache
export HF_HUB_CACHE="$PROJECT/.cq-kernel-cache"
export TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1
LOG=/home/eryx/cq-$(date +%m%d-%H%M%S)
# flock prevents races between two CQ launchers. Other launchers still require serial coordination.
exec flock -n /home/eryx/qwen3.5-opd/.cq-gpu.lock "$PYTHON" "$GUARD" \
  --log-dir "$LOG" --timeout "$GUARD_TIMEOUT" --interval 1 --min-mem-gib 5 --no-temperature-protection -- \
  "$PYTHON" -u -m cq_opd.trainer "$@"
