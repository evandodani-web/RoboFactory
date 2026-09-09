# RoboFactory Eval Leaderboard

Success rate on held-out seeds. Unless noted, **100 seeds** (1000–1099), checkpoint epoch **100**.

## LiftBarrier-rf — CLS study (150 demos)

| Rank | Study / config | Variant | Steps | SR | Successes | Run |
|------|----------------|---------|------|-----|-----------|-----|
| 1 | Baseline CLS-DP | `clsdp` (default) | default | **61.0%** | 61/100 | `LiftBarrier-rf_150_100_20260830_135720` |
| 2 | Factorized split latent (FG) | `clsdpfg` | 100 (ckpt default) | **55.0%** | 55/100 | `LiftBarrier-rf_clsdpfg_150_100_20260909_081648` |
| 3 | Deterministic latent (DET) | `clsdpdet` | 100 (ckpt default) | **49.0%** | 49/100 | `LiftBarrier-rf_clsdpdet_150_100_20260908_172328` |
| 4 | Flow matching (FM) | `clsdpfm` | 4 (ckpt default) | **23.0%** | 23/100 | `LiftBarrier-rf_clsdpfm_150_100_20260908_144039` |

## LiftBarrier-rf — earlier baseline (100 demos)

| Study / config | Variant | SR | Successes | Run |
|----------------|---------|-----|-----------|-----|
| Baseline CLS-DP | `clsdp` (default) | **37.0%** | 37/100 | `LiftBarrier-rf_100_100_20260829_024814` |

## LiftBarrier-rf — FG inference-step sweep (in progress)

25 seeds (1000–1024), variant `clsdpfg`. Videos under each run’s `videos/` folder.

| Steps | SR | Successes | Status | Run |
|------|-----|-----------|--------|-----|
| 4 | — | — | running | `LiftBarrier-rf_clsdpfg_s4_150_100_20260909_121807` |
| 8 | — | — | pending | — |
| 12 | — | — | pending | — |
| 24 | — | — | pending | — |

## Other tasks

| Task | Study / config | Variant | SR | Notes | Run |
|------|----------------|---------|-----|-------|-----|
| ThreeRobotsStackCube-rf | CLS-DP | `clsdp` | 0.0% | Incomplete (11/100 seeds) | `ThreeRobotsStackCube-rf_clsdp_150_100_20260904_175651` |

## Notes

- **FG** = factorized self/team latent (`cls_dp_fg` / `clsdpfg`)
- **DET** = deterministic prior (`clsdpdet`)
- **FM** = flow-matching action head (`clsdpfm`)
- Run dirs live under `robofactory/eval_results/`
- Last updated: 2026-09-09
