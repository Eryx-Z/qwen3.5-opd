#!/usr/bin/env bash
set -euo pipefail
# 300 fresh on-policy updates, 2048-token responses. No dev evaluation inherited.
bash /home/eryx/qwen3.5-opd/scripts/run_gsm8k_k1_throughput100.sh \
    data.max_response_length=2048 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2560 \
    actor_rollout_ref.rollout.max_model_len=2561 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=2560 \
    distillation.teacher_models.teacher_model.inference.max_model_len=2561 \
    trainer.experiment_name=gsm8k_k1_300_2048 \
    trainer.total_training_steps=300 \
    trainer.total_epochs=2 \
    "$@"
