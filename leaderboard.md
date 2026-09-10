# RoboFactory Eval Leaderboard

Success rate on held-out seeds. Unless noted, **100 seeds** (1000–1099), checkpoint epoch **100**, **150 demos**.

## LiftBarrier-rf — main results (100 seeds)

| Rank | Study / config | Variant | Inference steps | SR | Successes | Run |
|------|----------------|---------|-----------------|-----|-----------|-----|
| 1 | Flow matching (FM) | `clsdpfm` | **30** (euler) | **67.0%** | 67/100 | `LiftBarrier-rf_clsdpfm_s30_150_100_20260909_163941` |
| 2 | Study B H25 | `clsdph25` | 100 (DDPM default) | **66.0%** | 66/100 | `LiftBarrier-rf_clsdph25_150_100_20260910_033614` |
| 3 | Study B — baseline CLS-DP | `clsdp` (default) | 100 (DDPM default) | **61.0%** | 61/100 | `LiftBarrier-rf_150_100_20260830_135720` |
| 4 | Factorized split latent (FG) | `clsdpfg` | 100 (DDPM default) | **55.0%** | 55/100 | `LiftBarrier-rf_clsdpfg_150_100_20260909_081648` |
| 5 | Deterministic latent (DET) | `clsdpdet` | 100 (DDPM default) | **49.0%** | 49/100 | `LiftBarrier-rf_clsdpdet_150_100_20260908_172328` |
| 5 | Flow matching (FM) | `clsdpfm` | **50** (euler) | **49.0%** | 49/100 | `LiftBarrier-rf_clsdpfm_s50_150_100_20260910_002518` |
| 7 | Flow matching (FM) | `clsdpfm` | 4 (ckpt default) | **23.0%** | 23/100 | `LiftBarrier-rf_clsdpfm_150_100_20260908_144039` |

## LiftBarrier-rf — FM flow-step sweep (25 seeds, 1000–1024)

Euler flow matching (`clsdpfm`).

| Steps | SR | Successes | Run |
|------|-----|-----------|-----|
| 4 | **20.0%** | 5/25 | `LiftBarrier-rf_clsdpfm_s4_150_100_20260909_131802` |
| 8 | **28.0%** | 7/25 | `LiftBarrier-rf_clsdpfm_s8_150_100_20260909_135602` |
| 12 | **40.0%** | 10/25 | `LiftBarrier-rf_clsdpfm_s12_150_100_20260909_142601` |
| 24 | **60.0%** | 15/25 | `LiftBarrier-rf_clsdpfm_s24_150_100_20260909_145117` |

## LiftBarrier-rf — earlier baseline (100 demos)

| Study / config | Variant | SR | Successes | Run |
|----------------|---------|-----|-----------|-----|
| Baseline CLS-DP | `clsdp` (default) | **37.0%** | 37/100 | `LiftBarrier-rf_100_100_20260829_024814` |

## Notes

- **Study B** = baseline stochastic CLS-DP (`clsdp`, DDPM, sampled `z`)
- **Study B H25** = study B with action horizon 25 (`clsdph25`)
- **FM** = study B prior + flow-matching action head (`clsdpfm`, euler)
- **DET** / **FG** = deterministic / factorized latent variants
- Videos for recent runs: each run dir’s `videos/` folder
- Run dirs: `robofactory/eval_results/`
- Last updated: 2026-09-10
