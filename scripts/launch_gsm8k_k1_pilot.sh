#!/usr/bin/env bash
# Pilot supervisor. User requested temperature protection disabled; telemetry remains.
set -euo pipefail
PROJECT=/home/eryx/qwen3.5-opd
STAMP=$(date +%Y%m%d_%H%M%S)
export PILOT_DIR=${PILOT_DIR:-$PROJECT/runs/gsm8k_k1_pilot_$STAMP}
RUN_DIR=${RUN_DIR:-$HOME/gp-$STAMP}
printf 'Pilot output: %s\nGuard logs: %s\n' "$PILOT_DIR" "$RUN_DIR"
exec /home/eryx/verl/.venv/bin/python \
    /home/eryx/verl/examples/on_policy_distillation_trainer/diagnostics/guard.py \
    --log-dir "$RUN_DIR" --timeout 10800 --interval 1 --min-mem-gib 5 \
    --no-temperature-protection -- \
    bash "$PROJECT/scripts/run_gsm8k_k1_pilot.sh"
