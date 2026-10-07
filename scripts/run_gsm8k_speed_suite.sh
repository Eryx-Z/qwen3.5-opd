#!/usr/bin/env bash
set -euo pipefail
: "${SPEED_ROOT:?}" "${SPEED_DATA:?}" "${SPEED_GUARD_ROOT:?}"
index=0
PROJECT=/home/eryx/qwen3.5-opd
PY=/home/eryx/verl/.venv/bin/python
for SPEED_CASE in eager16 eager32 eager32_workers4 graph32; do
  export SPEED_CASE SPEED_DATA
  export PILOT_DIR="$SPEED_ROOT/$SPEED_CASE/output"
  index=$((index + 1))
  RUN="$SPEED_GUARD_ROOT/$index"
  mkdir -p "$RUN"
  ln -s "$RUN" "$SPEED_ROOT/$SPEED_CASE/guard"
  echo "$(date -Is) Starting $SPEED_CASE"
  rc=0
  "$PY" /home/eryx/verl/examples/on_policy_distillation_trainer/diagnostics/guard.py \
    --log-dir "$RUN" --timeout inf --interval 1 --min-mem-gib 5 --no-temperature-protection -- \
    bash "$PROJECT/scripts/run_gsm8k_speed_case.sh" || rc=$?
  echo "$(date -Is) Finished $SPEED_CASE guard_exit=$rc"
  "$PY" - "$RUN/outcome.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
if not x.get('cleanup',{}).get('ok',False):
    raise SystemExit('Cleanup failed; stop suite, do not overlap engines')
PY
done
echo "$(date -Is) Suite finished"
