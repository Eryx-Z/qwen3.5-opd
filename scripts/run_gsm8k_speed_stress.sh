#!/usr/bin/env bash
# Artificial fixed-token stress test only, NOT an algorithm/quality experiment.
set -euo pipefail
: "${STRESS_ROOT:?}" "${SPEED_DATA:?}"
PY=/home/eryx/verl/.venv/bin/python
PROJECT=/home/eryx/qwen3.5-opd
index=0
for SPEED_CASE in eager32 graph32; do
  index=$((index + 1))
  export SPEED_CASE SPEED_DATA PILOT_DIR="$STRESS_ROOT/$SPEED_CASE/output"
  RUN="$HOME/gz-$(date +%H%M%S)-$index"
  mkdir -p "$STRESS_ROOT/$SPEED_CASE" "$RUN"
  ln -s "$RUN" "$STRESS_ROOT/$SPEED_CASE/guard"
  extra=(actor_rollout_ref.rollout.ignore_eos=True trainer.total_training_steps=2 trainer.save_freq=2)
  bash "$PROJECT/scripts/run_gsm8k_speed_case.sh" "${extra[@]}" --cfg job > "$STRESS_ROOT/$SPEED_CASE/resolved-config.yaml"
  echo "$(date -Is) Starting fixed-token $SPEED_CASE"
  rc=0
  "$PY" /home/eryx/verl/examples/on_policy_distillation_trainer/diagnostics/guard.py \
    --log-dir "$RUN" --timeout 1800 --interval 1 --min-mem-gib 5 --no-temperature-protection -- \
    bash "$PROJECT/scripts/run_gsm8k_speed_case.sh" "${extra[@]}" || rc=$?
  echo "$(date -Is) Finished $SPEED_CASE guard_exit=$rc"
  "$PY" - "$RUN/outcome.json" <<'PY'
import sys,json
x=json.load(open(sys.argv[1]))
if not x.get('cleanup',{}).get('ok'):raise SystemExit('Stop: cleanup failed')
PY
done
echo "$(date -Is) Fixed-token suite finished"
