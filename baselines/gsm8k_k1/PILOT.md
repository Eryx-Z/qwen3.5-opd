# 200-step pilot

Started under independent guard: `/home/eryx/gp-184806`.
Output: `/home/eryx/qwen3.5-opd/runs/gsm8k_k1_pilot_200`.
This first pilot stopped during initial evaluation before any training updates: guard reason `thermal_limit`, ACPI 75.5 C, child exit 143, cleanup OK. Available memory at stop was 48.62 GiB. No completed paired dev evaluation or training result.

- Fresh original Student checkpoint and fresh LoRA, not continuation of the 20-step smoke.
- Same accepted sampled reverse-KL k1 / PPO-style policy-gradient surrogate, language-only LoRA and resident engines.
- 200 optimizer steps, batch 2, one current-policy response per prompt; train split has 6000 questions, seed 42. Maximum response length 1024 (previous smoke 256); prompt cap 512. Actual maximum train prompt length checked: 238 tokens.
- Fixed 64-question dev subset (first 64 rows of frozen seed-42 dev split). Greedy n=1 initial and final evaluation with identical 1024-token budget. Dev IDs and parquet SHA256 in `pilot-data-manifest.json`. No test use.
- Save every 50 steps, keep latest two checkpoints. Separate output path preserves smoke evidence.
- First attempt used a 10800-second maximum, 20 GiB available-memory floor and 75 C ACPI maximum. After this stop, the user requested removal of temperature protection. The new launcher `scripts/launch_gsm8k_k1_pilot.sh` explicitly uses `--no-temperature-protection`: temperature remains telemetry only, including sensor failures; memory protection, timeout and process cleanup remain active. The shared guard retains its original default for unrelated runs. No new run was started as part of this protection change.
- Launch script: `scripts/run_gsm8k_k1_pilot.sh`, requires `PILOT_DIR` and guard environment. Refuses existing output artifacts rather than overwriting. Configuration: `resolved-pilot-config.yaml`.

To inspect remotely:

```bash
tail -n 20 /home/eryx/gp-184806/smoke.log
# Terminal status, only available after guard finishes:
/home/eryx/verl/.venv/bin/python -m json.tool /home/eryx/gp-184806/outcome.json
```

Acceptance requires step 200, finite gradients/losses, checkpoint integrity, paired dev scores and truncation rates, guard exit 0 and clean process teardown. A 64-question single-seed comparison is preliminary, not a statistically robust quality claim. Do not launch an overlapping run. Start a fresh run with `bash scripts/launch_gsm8k_k1_pilot.sh`; it allocates fresh timestamped log/output directories and disables temperature protection as requested. Protection-change tests: 19 guard/allocation tests passed.
