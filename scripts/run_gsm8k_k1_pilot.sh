#!/usr/bin/env bash
# Fresh 200-step pilot; reuse the accepted single-GPU LoRA lifecycle.
set -euo pipefail
PROJECT=/home/eryx/qwen3.5-opd
: "${PILOT_DIR:?Set PILOT_DIR to a fresh absolute output directory}"
if [[ "$*" != *"--cfg"* ]]; then
    : "${VERL_OPD_GUARD_RUN:?Training must run under diagnostics/guard.py}"
    mkdir -p "$PILOT_DIR"
    if [[ -e "$PILOT_DIR/checkpoints" || -e "$PILOT_DIR/rollouts" || -e "$PILOT_DIR/validation" ]]; then
        echo "Refusing to overwrite an existing pilot" >&2
        exit 1
    fi
fi
bash "$PROJECT/scripts/run_gsm8k_k1.sh" \
    "data.train_files=['$PROJECT/data/gsm8k/train.parquet']" \
    "data.val_files=['$PROJECT/data/gsm8k/pilot_dev.parquet']" \
    data.seed=42 \
    data.val_batch_size=2 \
    data.validation_shuffle=False \
    data.max_response_length=1024 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1536 \
    actor_rollout_ref.rollout.max_model_len=1537 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1536 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=1536 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    distillation.teacher_models.teacher_model.inference.max_model_len=1537 \
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=1536 \
    trainer.experiment_name=gsm8k_k1_lora_pilot_200 \
    trainer.total_training_steps=200 \
    trainer.val_before_train=True \
    trainer.test_freq=200 \
    trainer.save_freq=50 \
    trainer.max_actor_ckpt_to_keep=2 \
    "trainer.default_local_dir=$PILOT_DIR/checkpoints" \
    "trainer.rollout_data_dir=$PILOT_DIR/rollouts" \
    "trainer.validation_data_dir=$PILOT_DIR/validation" \
    "$@"
