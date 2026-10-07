# Integration accepted

Successful run: `/home/eryx/gk-183212`. Full evidence: `acceptance-proof.json`, `step-metrics.jsonl`, `resolved-smoke-config.yaml`.

- 20/20 optimizer steps completed on 40 official GSM8K train examples.
- Finite losses and positive finite gradient norms at every step.
- Checkpoint saved at `runs/gsm8k_k1_smoke/checkpoints/global_step_20` with model, optimizer, scheduler/extra state, data state, tokenizer/config and LoRA metadata.
- Checkpoint audit: 372 language-only LoRA tensors, all finite FP32; all 186 initially zero B tensors nonzero. All 372 optimizer states at step 20.
- Independent guard: child exit 0, cleanup OK, no remaining owned processes. GPU process list empty; available system memory recovered to 118 GiB.
- Minimum available memory 25.28 GiB; maximum ACPI reading 72.2 C; total guard interval about 564 seconds. Original 20 GiB / 75 C / 600 sec thresholds retained.
- 109 existing/new verl placement/lifecycle/guard tests and 13 project tests passed.
- Test set not evaluated. Validation was disabled. 80% of training responses reached the 256-token cap. This is not evidence of GSM8K accuracy improvement.

## Fix

The first run `/home/eryx/gk-182304` failed during LoRA student level-1 sleep (`Memory usage increased after sleeping`), before updates. Level-1 sleep backs up frozen weights to CPU, which shares physical memory with GPU on Spark. The assertion measures global free memory, not isolated worker allocations; its exact numerical trigger was not instrumented.

A config-only attempt `/home/eryx/gk-182909` was rejected because the existing shared resource-pool policy required all students to sleep.

The accepted configuration keeps the student rollout resident (`free_cache_engine=false`, `enable_sleep_mode=false`). The resource-pool policy now allows this only for unmerged LoRA adapters; full-parameter restrictions remain unchanged. Existing adapter synchronization still clears caches after updating weights. No vLLM package edits and no bypassed assertions or safety limits.

`resident-lora.patch` captures the incremental changes to the previously recorded verl working tree and their regression tests. It is already applied; reverse dry-run checked. Existing environment snapshot is preserved.

## Next experiment (not started)

Longer response budget, paired initial/final dev evaluation, and a 200-step pilot. Use fresh output paths and an explicitly appropriate runtime limit while retaining memory/temperature protection. Current checkpoint resume/load in a training process has not yet been exercised; checkpoint tensor/optimizer integrity was checked on CPU.
