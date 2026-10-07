#!/usr/bin/env bash
# GSM8K integration smoke: sampled reverse-KL k1 + vanilla policy gradient, language LoRA.
set -euo pipefail
PROJECT=/home/eryx/qwen3.5-opd
cd /home/eryx/verl
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PROJECT:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VERL_USE_UV=0 DEVICE=gpu INFER_BACKEND=vllm
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT=/home/eryx/verl-install/opd-preflight/vllm-cache
export STUDENT_MODEL=/home/eryx/models/Qwen3.5-2B
export TEACHER_MODEL=/home/eryx/models/Qwen3.5-35B-A3B-FP8
export TRAIN_FILE="$PROJECT/data/gsm8k/smoke_train.parquet"
export VAL_FILE="$PROJECT/data/gsm8k/smoke_dev.parquet"
export NNODES=1 NGPUS_PER_NODE=1 TEACHER_WORLD_SIZE=1
export TRAIN_BATCH_SIZE=2 PPO_MINI_BATCH_SIZE=2
export MAX_PROMPT_LENGTH=512 MAX_RESPONSE_LENGTH=256 PPO_MAX_TOKEN_LEN_PER_GPU=768
export ROLLOUT_TP=1 TEACHER_TP=1 TEACHER_EP=1
export ROLLOUT_GPU_MEM_UTIL=0.15 TEACHER_GPU_MEM_UTIL=0.35
export TOTAL_EPOCHS=1 SAVE_FREQ=20 TEST_FREQ=-1
export USE_POLICY_GRADIENT=True DISTILLATION_LOSS_MODE=k1 ACTOR_LR=1e-6
export PROJECT_NAME=gsm8k_sampled_reverse_kl EXPERIMENT_NAME=k1_lora_smoke
TARGETS=$(python3 -c "import json; print(json.dumps(json.load(open('$PROJECT/baselines/gsm8k_k1/lora_manifest.json'))['target_modules'],separators=(',', ':')))")
bash examples/on_policy_distillation_trainer/run_qwen3_5_2b_fsdp.sh \
    distillation.share_student_resource_pool=True \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    "actor_rollout_ref.model.target_modules=$TARGETS" \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    'actor_rollout_ref.actor.optim.override_optimizer_config={foreach:False}' \
    actor_rollout_ref.actor.ppo_epochs=1 \
    reward.num_workers=1 \
    "reward.custom_reward_function.path=$PROJECT/baseline/verifier.py" \
    reward.custom_reward_function.name=compute_score \
    distillation.distillation_loss.loss_max_clamp=null \
    distillation.distillation_loss.log_prob_min_clamp=null \
    distillation.teacher_models.teacher_model.inference.free_cache_engine=False \
    +distillation.teacher_models.teacher_model.inference.enable_sleep_mode=False \
    distillation.teacher_models.teacher_model.inference.enforce_eager=True \
    distillation.teacher_models.teacher_model.inference.max_num_seqs=2 \
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=768 \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.kv_cache_memory_bytes=536870912 \
    '+distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.limit_mm_per_prompt={image:0,video:0}' \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.skip_mm_profiling=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    +actor_rollout_ref.rollout.enable_sleep_mode=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_seqs=2 \
    actor_rollout_ref.rollout.max_num_batched_tokens=768 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.generation_config=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=268435456 \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt={image:0,video:0}' \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.skip_mm_profiling=True \
    '+data.apply_chat_template_kwargs={enable_thinking:False}' \
    data.dataloader_num_workers=0 \
    trainer.total_training_steps=20 \
    trainer.logger='["console"]' \
    trainer.resume_mode=disable \
    "trainer.default_local_dir=$PROJECT/runs/gsm8k_k1_smoke/checkpoints" \
    "trainer.rollout_data_dir=$PROJECT/runs/gsm8k_k1_smoke/rollouts" \
    +ray_kwargs.ray_init.object_store_memory=268435456 \
    +ray_kwargs.ray_init.include_dashboard=False \
    ray_kwargs.ray_init.num_cpus=12 \
    "$@"
