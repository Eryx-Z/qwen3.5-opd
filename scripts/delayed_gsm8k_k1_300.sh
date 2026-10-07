#!/usr/bin/env bash
set -euo pipefail
: "${RUN_DIR:?Set a fresh guard log directory}"
: "${PILOT_DIR:?Set a fresh training output directory}"
export PILOT_DIR
sleeper=
trap 'if [[ -n "$sleeper" ]]; then kill "$sleeper" 2>/dev/null || true; fi; exit 143' TERM INT
printf 'Scheduled at %s; delay 3600 seconds\n' "$(date -Is)"
printf 'Cancel before launch: touch %s/CANCEL\n' "$RUN_DIR"
sleep 3600 &
sleeper=$!
wait "$sleeper"
sleeper=
if [[ -e "$RUN_DIR/CANCEL" ]]; then
    echo 'Cancelled before launch.'
    exit 0
fi
printf 'Starting at %s\n' "$(date -Is)"
# argparse accepts positive infinity; elapsed >= infinity is never true.
# Memory monitoring, conflict checks and owned-process cleanup stay enabled.
exec /home/eryx/verl/.venv/bin/python \
    /home/eryx/verl/examples/on_policy_distillation_trainer/diagnostics/guard.py \
    --log-dir "$RUN_DIR" --timeout inf --interval 1 --min-mem-gib 5 \
    --no-temperature-protection -- \
    bash /home/eryx/qwen3.5-opd/scripts/run_gsm8k_k1_300.sh
