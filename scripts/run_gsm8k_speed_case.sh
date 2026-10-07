#!/usr/bin/env bash
set -euo pipefail
: "${SPEED_CASE:?}" "${SPEED_DATA:?}" "${PILOT_DIR:?}"
extra=()
case "$SPEED_CASE" in
  eager16) extra+=(actor_rollout_ref.rollout.max_num_seqs=16) ;;
  eager32) extra+=(actor_rollout_ref.rollout.max_num_seqs=32) ;;
  eager32_workers4) extra+=(actor_rollout_ref.rollout.max_num_seqs=32 actor_rollout_ref.rollout.agent.num_workers=4) ;;
  graph32) extra+=(+distillation.allow_student_cuda_graph=True actor_rollout_ref.rollout.max_num_seqs=32 actor_rollout_ref.rollout.enforce_eager=False 'actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32]') ;;
  *) echo "Unknown case $SPEED_CASE" >&2; exit 1 ;;
esac
bash /home/eryx/qwen3.5-opd/scripts/run_gsm8k_k1_300.sh \
  "data.train_files=['$SPEED_DATA']" \
  trainer.total_training_steps=5 trainer.total_epochs=1 \
  trainer.save_freq=5 trainer.experiment_name="speed_$SPEED_CASE" \
  "${extra[@]}" "$@"
