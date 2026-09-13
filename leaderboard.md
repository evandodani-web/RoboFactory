# RoboFactory Eval Leaderboard

Success rate on held-out seeds. Unless noted, **100 seeds** (1000–1099), checkpoint epoch **100**, **150 demos**.

## LiftBarrier-rf — main results (100 seeds)


| Rank | Study / config                     | Variant           | Inference steps    | SR        | Successes | Run                                                  |
| ---- | ---------------------------------- | ----------------- | ------------------ | --------- | --------- | ---------------------------------------------------- |
| 1    | Flow matching (FM)                 | `clsdpfm`         | **30** (euler)     | **67.0%** | 67/100    | `LiftBarrier-rf_clsdpfm_s30_150_100_20260909_163941` |
| 2    | Study B H25                        | `clsdph25`        | 100 (DDPM default) | **66.0%** | 66/100    | `LiftBarrier-rf_clsdph25_150_100_20260910_033614`    |
| 3    | Study B-FM-H25 (noclamp)           | `clsdpfmh25`      | **30** (euler)     | **63.0%** | 63/100    | `LiftBarrier-rf_clsdpfmh25_noclamp_150_100_20260913_105937` |
| 4    | Study B — baseline CLS-DP          | `clsdp` (default) | 100 (DDPM default) | **61.0%** | 61/100    | `LiftBarrier-rf_150_100_20260830_135720`             |
| 5    | Factorized split latent (FG) + DET | `clsdpfg`         | 100 (DDPM default) | **55.0%** | 55/100    | `LiftBarrier-rf_clsdpfg_150_100_20260909_081648`     |
| 6    | Deterministic latent (DET)         | `clsdpdet`        | 100 (DDPM default) | **49.0%** | 49/100    | `LiftBarrier-rf_clsdpdet_150_100_20260908_172328`    |
| 6    | Flow matching (FM)                 | `clsdpfm`         | **50** (euler)     | **49.0%** | 49/100    | `LiftBarrier-rf_clsdpfm_s50_150_100_20260910_002518` |
| 6    | Combo B-FG-FM-H25-uni (noclamp)    | `clsdpbfgfmh25uni`| **30** (euler)     | **49.0%** | 49/100    | `LiftBarrier-rf_clsdpbfgfmh25uni_noclamp_150_100_20260912_072852` |
| 9    | Flow matching v2 (FM-v2)           | `clsdpfmv2`       | **30** (euler)     | **46.0%** | 46/100    | `LiftBarrier-rf_clsdpfmv2_150_100_20260910_202240`  |
| 10   | Combo B-FG-FM-H25 Beta (noclamp)   | `clsdpbfgfmh25`   | **30** (euler)     | **38.0%** | 38/100    | `LiftBarrier-rf_clsdpbfgfmh25_noclamp_150_100_20260911_160350` |
| 11   | Flow matching (FM)                 | `clsdpfm`         | 4 (ckpt default)   | **23.0%** | 23/100    | `LiftBarrier-rf_clsdpfm_150_100_20260908_144039`     |




## LiftBarrier-rf — FM flow-step sweep (25 seeds, 1000–1024)

Euler flow matching (`clsdpfm`).


| Steps | SR        | Successes | Run                                                  |
| ----- | --------- | --------- | ---------------------------------------------------- |
| 4     | **20.0%** | 5/25      | `LiftBarrier-rf_clsdpfm_s4_150_100_20260909_131802`  |
| 8     | **28.0%** | 7/25      | `LiftBarrier-rf_clsdpfm_s8_150_100_20260909_135602`  |
| 12    | **40.0%** | 10/25     | `LiftBarrier-rf_clsdpfm_s12_150_100_20260909_142601` |
| 24    | **60.0%** | 15/25     | `LiftBarrier-rf_clsdpfm_s24_150_100_20260909_145117` |




## LiftBarrier-rf — FM-v2 flow-step / denoise sweep (50 seeds, 1000–1049)

Euler flow matching (`clsdpfmv2`, Beta sigma + clamp from ckpt). Same harder seed half as other 50-seed ablations. Headline 30-step result in the main table is 100 seeds.


| Steps | SR        | Successes | Run                                                   |
| ----- | --------- | --------- | ----------------------------------------------------- |
| 4     | **22.0%** | 11/50     | `LiftBarrier-rf_clsdpfmv2_s4_150_100_20260910_232435` |
| 8     | **36.0%** | 18/50     | `LiftBarrier-rf_clsdpfmv2_s8_150_100_20260911_002930` |
| 16    | **40.0%** | 20/50     | `LiftBarrier-rf_clsdpfmv2_s16_150_100_20260911_012004` |
| 20    | **30.0%** | 15/50     | `LiftBarrier-rf_clsdpfmv2_s20_150_100_20260911_021434` |
| 30    | **44.0%** | 22/50*    | `LiftBarrier-rf_clsdpfmv2_150_100_20260910_202240` (same half of 100-seed run) |

*Full 100-seed SR at 30 steps is **46.0%** (46/100).

**Read:** rises 4→16, dips at 20, then recovers — best on this half at **30 (44%)** / full protocol **46%**. Not a clean "more steps → better" curve (unlike FM-v1).




## LiftBarrier-rf — Combo B-FG-FM-H25 flow-step sweep (50 seeds, 1000–1049)

Euler flow matching (`clsdpbfgfmh25`, noclamp, `max_steps=65`). Headline 30-step result above is 100 seeds.


| Steps | SR        | Successes | Run                                                              |
| ----- | --------- | --------- | ---------------------------------------------------------------- |
| 4     | **46.0%** | 23/50     | `LiftBarrier-rf_clsdpbfgfmh25_s4_noclamp_150_100_20260911_170951` |
| 8     | **52.0%** | 26/50     | `LiftBarrier-rf_clsdpbfgfmh25_s8_noclamp_150_100_20260911_174025` |
| 16    | **40.0%** | 20/50     | `LiftBarrier-rf_clsdpbfgfmh25_s16_noclamp_150_100_20260911_180409` |
| 20    | **46.0%** | 23/50     | `LiftBarrier-rf_clsdpbfgfmh25_s20_noclamp_150_100_20260911_183532` |
| 30    | **38.0%** | 19/50*    | `LiftBarrier-rf_clsdpbfgfmh25_noclamp_150_100_20260911_160350` (same half of 100-seed run) |

*Full 100-seed SR at 30 steps is also **38.0%** (38/100).

**Read:** peak at **8 steps (52%)**; more denoising does **not** help — 16/20/30 are flat-to-worse. Opposite shape from FM-v1 (more steps → better) and unlike FM-v2 (best at 30).



## LiftBarrier-rf — Combo B-FG-FM-H25-uni flow-step sweep (50 seeds, 1000–1049)

Euler flow matching (`clsdpbfgfmh25uni`, noclamp, `max_steps=65`). Headline 30-step result in the main table is 100 seeds.


| Steps | SR        | Successes | Run                                                                 |
| ----- | --------- | --------- | ------------------------------------------------------------------- |
| 16    | **42.0%** | 21/50     | `LiftBarrier-rf_clsdpbfgfmh25uni_s16_noclamp_150_100_20260912_132538` |
| 24    | **48.0%** | 24/50     | `LiftBarrier-rf_clsdpbfgfmh25uni_s24_noclamp_150_100_20260912_134356` |
| 30    | **44.0%** | 22/50*    | `LiftBarrier-rf_clsdpbfgfmh25uni_noclamp_150_100_20260912_072852` (same half of 100-seed run) |
| 40    | **42.0%** | 21/50     | `LiftBarrier-rf_clsdpbfgfmh25uni_s40_noclamp_150_100_20260912_140215` |
| 50    | **42.0%** | 21/50     | `LiftBarrier-rf_clsdpbfgfmh25uni_s50_noclamp_150_100_20260912_142318` |

*Full 100-seed SR at 30 steps is **49.0%** (49/100).

**Read:** peak at **24 steps (48%)** on this half; 16/40/50 are flat at 42%. Extra denoising past ~24 does **not** help (unlike FM-v1).



## LiftBarrier-rf — Combo schedule: uniform vs Beta (100 seeds, 30 Euler, noclamp)

Same B-FG-FM-H25 stack and shared `ctxbfgh25` priors; only Stage-2 `sigma_dist` differs. Both evals: `max_steps=65`, `--no-clamp-x1`, shift=1.0.


| Schedule | Variant | SR | Successes | Run |
| -------- | ------- | -- | --------- | --- |
| **uniform** | `clsdpbfgfmh25uni` | **49.0%** | 49/100 | `LiftBarrier-rf_clsdpbfgfmh25uni_noclamp_150_100_20260912_072852` |
| **Beta(1.5,1)** | `clsdpbfgfmh25` | **38.0%** | 38/100 | `LiftBarrier-rf_clsdpbfgfmh25_noclamp_150_100_20260911_160350` |

**Read:** uniform wins by **+11 pts** on the full protocol — same direction as FM-v1 (uniform) vs FM-v2 (Beta).



## LiftBarrier-rf — B-FM-H25 deciding run (100 seeds, 30 Euler, noclamp)

Monolithic Study B latent + flow @ h25 (`clsdpfmh25`). Same `*_ctxh25_*` priors as B-H25; Stage 2 only. Protocol: `max_steps=65`, `--no-clamp-x1`, shift=1.0. This splits the combo's 17-point deficit: near 66% indicts factorization; near 49% indicts flow@h25.


| Comparator | Variant | SR | vs B-FM-H25 (McNemar z) | Run |
| ---------- | ------- | -- | ----------------------- | --- |
| **B-FM-H25** | `clsdpfmh25` | **63.0%** | — | `LiftBarrier-rf_clsdpfmh25_noclamp_150_100_20260913_105937` |
| B-H25 (DDPM) | `clsdph25` | 66.0% | z=−0.63 (n.s.) | `LiftBarrier-rf_clsdph25_150_100_20260910_033614` |
| FM @ h8 | `clsdpfm` @30 | 67.0% | z=−0.59 (n.s.) | `LiftBarrier-rf_clsdpfm_s30_150_100_20260909_163941` |
| Study B | `clsdp` | 61.0% | z=+0.31 (n.s.) | `LiftBarrier-rf_150_100_20260830_135720` |
| Combo uni | `clsdpbfgfmh25uni` | 49.0% | z=+1.98 (sig.) | `LiftBarrier-rf_clsdpbfgfmh25uni_noclamp_150_100_20260912_072852` |

**Read:** flow@h25 **survives** — statistically tied with B-H25 and FM@h8. The combo's regression is factorization (or its interaction with the stack), not the flow head at h25.



## LiftBarrier-rf — clamp × schedule 2×2 (50 seeds, 1000–1049, 30 Euler steps)

Isolates whether FM-v2's drop (67% → 46%) is the clamp, the Beta schedule, or both.
FM-v1's saved config predates `clamp_x1`, so today's class default (1.0) turns clamp on.
FM-v2 saves `clamp_x1: 1.0`; unclamped cell uses `--no-clamp-x1`.


| Schedule \\ Clamp | unclamped | clamped |
| ----------------- | --------- | ------- |
| **uniform (FM v1)** | **67.0%** (100 seeds, historical) `clsdpfm` @30 | **50.0%** 25/50 `LiftBarrier-rf_clsdpfm_s30_clamp_150_100_20260911_101112` |
| **Beta (FM-v2)** | **46.0%** 23/50 `LiftBarrier-rf_clsdpfmv2_s30_noclamp_150_100_20260911_110637` | **46.0%** (100 seeds) `LiftBarrier-rf_clsdpfmv2_150_100_20260910_202240` |

**Read:** clamp on FM-v1 costs ~17 pts (67→50), so clamp hurts. Unclamping FM-v2 does **not** recover (stays 46%), so the Beta schedule is **not** exonerated — both knobs matter; schedule alone is enough to keep SR near the clamped FM-v2 floor.




## LiftBarrier-rf — FM-v2 inference knobs (50 seeds, 1000–1049, 30 Euler steps)

Same harder seed half. All keep Beta from the ckpt.


| Config | Clamp | Shift | SR | Successes | Run |
| ------ | ----- | ----- | -- | --------- | --- |
| FM-v2 default (100-seed headline) | on (ckpt) | 1.0 | **46.0%** | 46/100 | `LiftBarrier-rf_clsdpfmv2_150_100_20260910_202240` |
| FM-v2 noclamp | off | 1.0 | **46.0%** | 23/50 | `LiftBarrier-rf_clsdpfmv2_s30_noclamp_150_100_20260911_110637` |
| FM-v2 noclamp + shift 0.3 | off | **0.3** | **40.0%** | 20/50 | `LiftBarrier-rf_clsdpfmv2_s30_noclamp_shift0.3_150_100_20260911_143430` |

**Read:** `shift=0.3` did **not** help on this half (−6 pts vs noclamp).




## LiftBarrier-rf — earlier baseline (100 demos)


| Study / config  | Variant           | SR        | Successes | Run                                      |
| --------------- | ----------------- | --------- | --------- | ---------------------------------------- |
| Baseline CLS-DP | `clsdp` (default) | **37.0%** | 37/100    | `LiftBarrier-rf_100_100_20260829_024814` |




## Notes

- **Study B** = baseline stochastic CLS-DP (`clsdp`, DDPM, sampled `z`)
- **Study B H25** = study B with action horizon 25 (`clsdph25`)
- **Study B-FM-H25** = Study B prior @ h25 + flow Stage 2 (`clsdpfmh25`); deciding run for the combo regression — **63%**, tied with B-H25/FM; factorization indicted
- **FM** = study B prior + flow-matching action head (`clsdpfm`, euler)
- **FM-v2** = same as FM with Beta(1.5,1) sigma schedule + clamp + 30-step default (`clsdpfmv2`)
- **Combo B-FG-FM-H25** = Study B + FG + FM + horizon 25, Beta schedule (`clsdpbfgfmh25`); eval used `--no-clamp-x1`, `max_steps=65`, shift=1.0
- **Combo B-FG-FM-H25-uni** = same stack with **uniform** sigma (`clsdpbfgfmh25uni`); same priors/horizon; +11 pts vs Beta arm (49% vs 38%)
- **DET** / **FG** = deterministic / factorized latent variants
- Videos for recent runs: each run dir’s `videos/` folder
- Run dirs: `robofactory/eval_results/`
- **Policy:** every eval run (full protocol or ablation) gets a leaderboard row/section — no silent results.
- Last updated: 2026-09-13

