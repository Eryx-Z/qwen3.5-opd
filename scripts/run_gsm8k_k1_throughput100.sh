#!/usr/bin/env bash
set -euo pipefail
# Throughput-oriented 100 updates; fresh LoRA, no dev evaluation.
PROJECT=/home/eryx/qwen3.5-opd
bash "$PROJECT/scripts/run_gsm8k_k1_pilot.sh" \
    data.train_batch_size=32 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=4294967296 \
    distillation.teacher_models.teacher_model.inference.max_num_seqs=8 \
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=4096 \
    distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.kv_cache_memory_bytes=2147483648 \
    trainer.experiment_name=gsm8k_k1_throughput100 \
    trainer.total_training_steps=100 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    "$@"
