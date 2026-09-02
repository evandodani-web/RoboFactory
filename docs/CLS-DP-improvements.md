# CLS-DP v2: Proposed Improvements

Design doc for a stronger variant of CLS-DP. Collects the ideas from our research pass, ranked by
expected value.

Companions: [CLS-DP-replication-spec.md](CLS-DP-replication-spec.md) is the paper extraction,
[CLS-DP-implementation-notes.md](CLS-DP-implementation-notes.md) covers the baseline implementation
that already exists in this repo.

Nothing here is implemented yet. Confidence labels are honest: **high** means multiple papers agree
and the mechanism is clear, **medium** means one good result plus sound reasoning, **speculative**
means it follows from the diagnosis but nobody has published it.

---

## 1. CLS-DP in sixty seconds

Skip if you know it.

Several robot arms cooperate on one task. At deployment each arm runs **completely alone**: it sees
only its own camera and a shared text instruction. No shared overhead view, no exchanging joint
states, no communication of any kind. The problem is obvious — how do you coordinate with someone
you can't talk to and can barely see?

CLS-DP's answer is to train with information you throw away at deployment:

- **Stage 1 (contextualizer).** A conditional VAE. Its *posterior* gets privileged input: the future
  joint trajectories of **every** arm. Its *prior* gets only this arm's current camera frame plus the
  instruction. A KL term pulls them together. A decoder must reconstruct all arms' futures from just
  the latent plus this arm's own current state — so the latent is forced to carry teammate
  information. The posterior and decoder are then deleted.
- **Stage 2 (action-expert).** Freeze the prior. Sample a latent `z` from it, and condition a
  diffusion policy on `z` through cross-attention. `z` is the coordination signal.

One mechanism detail that matters below: the posterior mean is written as `prior mean + residual`, so
the prior mean **cancels out of the KL entirely**. The KL can only shrink the residual. The prior
mean is therefore trained purely by reconstruction — which is exactly what forces it to encode
teammate dynamics. That asymmetry is the heart of the method.

**Results:** 38% mean success over six RoboFactory tasks (2–4 arms), vs 20% for the best centralized
baseline and 9% for the same policy without the latent.

---

## 2. Where the baseline is weak

Everything in section 3 traces back to one of these.

| # | Weakness | How we know |
|---|---|---|
| W1 | `z` conflates "my own task progress" with "teammate dynamics" | The paper's own Fig. 5 notes the ablation also fails a subtask "requiring no inter-agent coordination" |
| W2 | `z` is resampled every control cycle from a **single frame**, with no coupling to the previous `z` | By construction; CoLA-Flow shows non-recurrent latents cause trajectory jitter |
| W3 | Each agent samples `z` **independently**, so they can commit to conflicting coordination hypotheses | Structural. Not noted in the paper |
| W4 | 100-step DDPM; latency compounds across agents | Table I; flow-matching policies are 7× faster at equal or better success |
| W5 | Evaluation locks every agent to the slowest one each cycle | `eval_multi_dp.py` computes `max_step` across agents and pads. Contradicts the decentralization claim |
| W6 | No smoothness objective anywhere; TOPP smoothing at eval is a band-aid | Repo code |
| W7 | Separate model per agent: no weight sharing, no cross-agent transfer, 150 demos each | Eq. 5 sums over per-agent parameters |
| W8 | No robustness testing (agent dropout, delay) despite decentralization being the whole premise | Absent from the paper. LatentToM makes this its headline advantage |
| W9 | No way to tell whether `z` covers multiple coordination modes or memorized one strategy | No multimodality metric exists |
| W10 | Take Photo is the worst task (8%), attributed to fine spatial precision; the policy backbone is ResNet-18 | Paper Sec. V-B.1 |

---

## 3. Ranked proposals

| Rank | Proposal | Targets | Confidence | Cost |
|---|---|---|---|---|
| **1** | Temporal-consistency loss on action deltas | W6, W2 | High | Low |
| **2** | Factorize `z` into self / teammate | W1 | Medium | Low |
| **3** | Fix deployment latent sampling | W3, W2 | Medium | Trivial |
| **4** | Multi-horizon delta targets in Stage 1 | W1 | High | Low |
| **5** | Flow-matching action expert | W4 | High | Medium |
| **6** | Asynchronous execution, drop the lockstep | W5 | Medium | Medium |
| **7** | Robustness suite | W8 | High | Low |
| **8** | Mode-coverage benchmark | W9 | Medium | Medium |
| **9** | Encoder upgrades | W10 | Medium | Low–Med |
| **10** | Shared backbone + agent conditioning | W7 | Medium | Medium |
| **11** | Latent-predictive (JEPA) auxiliary term | W1 | Medium | Medium |
| **12** | Task gradient into the contextualizer | — | Speculative | Medium |

---

### 1. Temporal-consistency loss on action deltas

**What.** Penalize the difference between the predicted action chunk's finite differences and the
ground truth's: `‖Δâ_t − Δa_t‖²`. Since state == action in RoboFactory, this is joint-velocity
matching.

**Why it's first.** Two independent papers converged on it. FlowWM adds exactly this to a world model
for +5.2% relative, at zero training overhead. PSG-JEPA supervises `Δq_{t,k} = q_{t+k} − q_t` and
finds removing it causes their largest planning drop (95.0 → 81.3). Meanwhile CLS-DP has *no*
smoothness objective at all and papers over jitter with TOPP at eval.

**Implementation note.** You do **not** need to unroll the sampler or compute a clean-sample estimate.
Finite differencing is linear, so it commutes through the noise parameterization. For DDPM
ε-prediction with `α = √ᾱ_t`, `σ = √(1−ᾱ_t)`:

```
‖Δx̂₀ − Δx₀‖²  =  (σ/α)² · ‖Δε − Δε̂‖²
```

So it's just a timestep-weighted finite-difference penalty on the epsilon residual. Closed form, one
extra term in `compute_loss`.

**Watch out.** That `(σ/α)²` factor vanishes at low noise. Any auxiliary loss applied at the clean-sample
estimate is dominated by the high-noise regime where that estimate is worthless. FlowWM compensates
with `λ(τ) = τ^γ`, and their ablation is dramatic: γ=1 gives 12.4 AP, γ=15 gives 20.9 — uniform
weighting was *worse than not doing it at all*. Reweight, or apply the loss to a fully unrolled sample.

---

### 2. Factorize `z` into self and teammate components

**What.** Split the 256-d latent into `[z_self ; z_team]`. Supervise `z_self` against this agent's own
future, `z_team` against the others'. Separate KL weights.

**Why.** One vector is currently doing two jobs, and the paper half-admits it. Everyone else working
on this problem factorizes: LatentToM keeps an *ego* embedding plus a *consensus* embedding and has
each agent decode the other's ego embedding; Dreaming of Others splits a Dreamer latent into
environment and teammate parts with a dedicated Theory-of-Mind head.

**Payoff beyond accuracy.** You get a clean ablation ("zero out `z_team`") and an interpretable
quantity. Right now you cannot separate "the latent helps me coordinate" from "the latent helps me do
my own job better."

**This is the most likely novel contribution of the set.**

---

### 3. Fix deployment latent sampling

**What.** Change how `z` is drawn at rollout. Default to the **prior mean** rather than sampling.

**Why this is a real bug and not a knob.** In single-agent generative policies, sampling gives you
diverse-but-valid behavior. Here, every agent samples *independently* with nothing coupling the draws.
On exactly the ambiguous dimensions that matter — who commits first — two agents can sample
conflicting hypotheses. **The sampling itself becomes a coordination failure mode.** Worse, the
posterior is widest precisely where the ambiguity is greatest.

**Options, in order of how much I'd trust them:**

1. **Take the mean.** All agents commit to the modal hypothesis. `latent_sample: false`.
2. **Shared random seed across agents.** Standard Dec-POMDP device (common randomness enables
   correlated equilibria) and not runtime communication — but it needs a caveat in any write-up.
3. **Sample once per episode or subtask, then hold.** Also fixes W2 for free.

Cheapest experiment in the doc. Run all four settings as an ablation table.

---

### 4. Multi-horizon delta targets in Stage 1

**What.** Stage 1 currently reconstructs 8 absolute future states at one horizon. Instead supervise
displacements `Δq_{t,k} = q_{t+k} − q_t` for `k = 1…8`. Same data, ~8× more supervision signal.

**Why.** PSG-JEPA argues a displacement target is strictly better than either absolute state or the
intervening action sequence: it's uniquely determined by the endpoints and has fixed dimension at
every horizon, whereas many action sequences can connect the same two configurations.

**Why it isn't redundant with the existing reconstruction**, even though absolute states are already
supervised. Supervising absolutes says nothing about whether the *errors* are correlated between
adjacent timesteps. With `q_t = 1.000` and `q_{t+1} = 1.010`, two decodes each carrying 1% absolute
error in opposite directions give `Δq̂ = −0.010` — right magnitude, wrong sign. Both endpoints look
excellent; the displacement is useless. Displacements between adjacent steps are small by
construction, so this is the normal case, not a corner case. PSG-JEPA puts it as: static grounding
"supervises each endpoint independently and does not directly optimize a pairwise readout of their
change."

**But do it with one latent, not a pair.** PSG-JEPA needs a latent *pair* because their latent is a
per-frame encoding — `q_t` comes from `z_t` and `q_{t+k}` from `z_{t+k}`. Ours is a whole-future
summary: the decoder already emits `q_{t+1} … q_{t+8}` from a single latent, so differencing that
output gets the same reweighting for free. A cross-latent pair head buys one extra thing —
correlating errors across *separate* prior invocations, i.e. consistency of the decoded belief from
one control cycle to the next — at the cost of a second prior pass (~2.5 GB extra activations at
batch 512). Worth it only if rollouts actually look jittery. See
[CLS-DP-variant-factorized-grounded.md](CLS-DP-variant-factorized-grounded.md) §5.3.

**Caveat.** For *teammates* you can't take a delta relative to their current state, because the
decoder never sees it. Predict teammate current state **and** displacement separately.

---

### 5. Flow-matching action expert

**What.** Replace the 100-step DDPM with flow matching; expect 4–10 inference steps.

**Why.** CoLA-Flow reports 7.5 ms per control step vs 55.9 ms for DP3 — 7.5× — with higher success.
E3Flow reports 7× with +3.12%. In single-agent that's convenience; here latency compounds across
agents and interacts with the synchronization problem in W5.

**Critical caveat.** Flow matching in *raw action space* is jittery — this is CoLA-Flow's central
finding. Do it in a latent action space, or pair it with proposal 1.

**Methodology warning.** If you swap the generative model, "38 vs 9" silently becomes "our gain plus
flow matching's gain." The `w/o CLS` ablation must be re-run as a flow policy too. **Keep the DDPM
action expert as a config toggle rather than deleting it.**

**Side effect that simplifies things:** at 5 steps, backpropagating through the whole sampler becomes
affordable. That kills the need for FlowWM's one-step projection trick, and with it the whole `τ^γ`
weighting-schedule tuning axis. (One-step projection is literally a single Euler step to `τ=1`, so a
flow straight enough for 5-step inference is straight enough that the projection and a real unroll
agree anyway.)

---

### 6. Asynchronous execution

**What.** Delete the `max_step` lockstep in the eval loop. Let each agent run on its own clock and
handle stale observations properly.

**Why.** The current loop makes every agent wait for the slowest, which quietly contradicts the
decentralization claim — genuinely independent agents don't synchronize. The async-VLA literature is
mature (RTC, TT-RTC, A2C2, REMAC, FutureRTC) and, as far as I can tell, **nobody has applied it to
multi-agent**, where chunk-boundary discontinuity interacts with coordination.

Starting points from the async survey: A2C2 is best under high delay, TT-RTC is the most robust
training-time method with zero inference overhead.

This is the largest genuinely open research gap on the list.

---

### 7. Robustness suite

**What.** Kill an agent mid-episode. Inject per-agent latency. Freeze one arm. Report degradation.

**Why.** LatentToM's headline claim is being "naturally robust to temporary robot failure or delays,
while a centralized policy may fail." CLS-DP never tests this despite decentralization being its
entire premise. It's the most persuasive argument for the approach and it's currently missing.

Cheap: runs on the existing eval harness.

---

### 8. Mode-coverage benchmark

**What.** A synthetic multi-agent task where valid coordination orderings are **enumerable** ("which
arm commits first"). Then measure, per FlowWM's Bouncing Shapes protocol:

- **Precision error** — are sampled predictions valid modes?
- **Recall error** — do samples cover all valid modes?
- **F1** — harmonic mean.

Plus best-of-N success on the real tasks.

**Why.** Today you cannot distinguish "`z` encodes coordination" from "`z` memorized the one strategy
in the demos." Since the demos come from a single scripted motion planner, the second is a live
worry. This is the instrument for that question, and it makes the multimodality argument for keeping
the CVAE testable rather than rhetorical.

---

### 9. Encoder upgrades

Three separable changes, cheapest first:

- **Intermediate-layer averaging.** Average encoder tokens from every third layer instead of taking
  the last. FlowWM does this citing Perception Encoder ("the best visual embeddings are not at the
  output of the network"). ~5 lines in `siglip_encoder.py`.
- **SigLIP → SigLIP 2** for the prior. Drop-in, keeps text alignment.
- **ResNet-18 → DINOv3** for the policy backbone. The evidence is split in a useful way: DINOv3 wins
  on fine-grained spatial precision (LARY, the JEPA-WM study), V-JEPA 2.1 wins on temporal
  anticipation. CLS-DP's worst task is bottlenecked on *spatial precision* specifically, so DINOv3
  targets the actual weakness.

---

### 10. Shared backbone + agent conditioning

**What.** One trunk for all agents, with a learned agent token, instead of N independent models.

**Why.** Cuts parameters roughly N×, pools N× the training data (you only have 150 demos), and opens
the door to generalizing to team sizes never seen in training — which is the scalability claim the
paper makes but never tests past 4 agents.

**Risk.** Eq. 5 sums over per-agent parameters, so this deviates from the paper. Worth it, but it
becomes a design choice to defend rather than a faithful reproduction.

---

### 11. Latent-predictive (JEPA) auxiliary term

**What.** Alongside the joint-state targets, predict teammates' future *visual features*.

**Why the framing matters.** The reflexive 2026 advice is "reconstruction is dead, predict in latent
space." That advice is about **pixel** reconstruction. CLS-DP reconstructs 8-d proprioceptive
vectors, which is much closer to what PSG-JEPA calls *physical state grounding* — and PSG-JEPA's whole
result is that pure latent forward-prediction **under-determines** robot state, so you should add
proprioceptive grounding back.

So: **add** the latent-predictive term, don't substitute it. CLS-DP is already leakage-free in
VLA-JEPA's sense (the prior never sees the future; it's only a target), so this composes without a
redesign.

---

### 12. Task gradient into the contextualizer

**What.** Let Stage-2 action loss flow back into the frozen prior with a small weight, instead of a
hard stop-gradient. A Stage-3 fine-tune.

**Why speculative.** FlowWM does this with a frozen detector and gets a real but modest gain, and
notes it's "bounded by the limited robustness of the frozen detector to noisy predicted latents." Our
failure mode is worse: the prior could drift toward whatever helps action prediction and *lose* the
coordination content. Keep the KL on, weight it small, and watch the Stage 1 gate as a canary.

---

## 4. Decided against

| Idea | Why not |
|---|---|
| **Remove the CVAE** | It isn't a compression device, it's a distillation mechanism, and the learned per-dimension `σ_p` is load-bearing: under partial observability some coordination dimensions are genuinely unknowable from one frame, and a deterministic L2 has no way to say so. It forces mode-averaging instead. FlowWM's Appendix A proves deterministic ℓ1/ℓ2 predictors converge to the conditional mean, "which corresponds to no valid future." Keep the distribution. |
| **Discrete latents (VQ / FSQ)** | FSQ genuinely fixes codebook collapse and GPC found CVAE priors collapse where FSQ didn't. But the residual parameterization with zero-init already starts the KL at exactly 0, which is a strong anti-collapse device, and the Stage 1 gate detects collapse directly. Revisit only if the gate fails. The one attraction: discrete "coordination modes" would be far more interpretable. |
| **SE(3) equivariance** | Real gains (E3Flow, SDP) but a large rewrite, and RoboFactory's fixed tabletop barely exercises SE(3) generalization. |
| **FlowWM's shifted timestep schedule** | Their single biggest jump (14.3 → 19.1 AP), but it compensates for *high-dimensional* latents. Our action tensor is 8×8 = 64 values. The principle predicts our optimal shift is near zero. Revisit only if the contextualizer starts predicting visual features. |
| **FlowWM's wide projection head** | Same reason. Their rule is "head width ≥ latent dim"; we're already 768 → 256. Their useful *negative* result: depth 2→8 changed nothing, so don't over-parameterize depth either. |
| **One-step projection** | Superseded. Not needed for linear objectives (see proposal 1), and at 5-step flow a real unroll is both affordable and more faithful. |

Free negative results from FlowWM, so nobody re-runs them: ℓ1 instead of ℓ2 for the flow objective
*degraded* performance, and reduced-rank noise (sample once, tile across time) consistently
underperformed.

---

## 5. Suggested sequencing

**Phase 1 — cheap, independent, no architecture change.** Proposals 1, 3, 4, 7. All are small edits
plus eval runs. Phase 1 alone would produce a defensible short paper if the numbers move.

**Phase 2 — the contribution.** Proposal 2 (factorized latent), validated with proposal 8
(mode-coverage benchmark). These pair naturally: the benchmark is how you prove the factorization did
something.

**Phase 3 — systems.** Proposals 5 and 6 together. Flow matching makes async execution tractable and
async execution is what makes flow matching's speed matter.

**Phase 4 — opportunistic.** 9, 10, 11, 12 as time allows.

---

## 6. Open questions

1. Does `z` actually carry multimodal coordination hypotheses, or one memorized strategy? The demos
   come from a single scripted planner, so there may be only one mode in the data to begin with. If
   so, proposals 3 and 8 both change character, and the case for a *stochastic* latent weakens.
2. How much of the 38% vs 9% gap survives once the ablation gets the same temporal-consistency loss?
   Some of the gain may be smoothness rather than coordination.
3. Is per-agent separate training actually necessary, or an artifact of small scale? Proposal 10
   tests this.
4. Does the factorized latent stay factorized, or does `z_self` quietly absorb teammate information?
   Needs a probe, not just an ablation. Concretely: train a head on a **frozen** `z_self` to recover
   teammate state. If it succeeds, the split is cosmetic.
5. **Is the prior alone actually sufficient?** Stage 1 never runs the decoder on the prior's latent
   by itself — every reconstruction uses `z_prior + residual`. But deployment has no residual. The
   Stage 1 gate inherits this, so it can pass while the prior alone is useless. Logging a prior-only
   reconstruction costs one extra decoder pass and no gradient, and it tests the mechanism the whole
   method rests on. This is arguably the highest-value cheap experiment on the list.

---

## 7. Evidence

| Paper | Used for |
|---|---|
| [CLS-DP](https://arxiv.org/abs/2606.22982) | The baseline |
| [PSG-JEPA](https://arxiv.org/html/2608.06799) | Multi-horizon delta targets; keep physical grounding |
| [VLA-JEPA](https://arxiv.org/html/2602.10098v1) | Leakage-free latent prediction |
| [Reconstruction or Semantics?](https://cvpr26wmas.github.io/papers/nilaksh26reconstruction.pdf) | Semantic > reconstruction latents for policy |
| [FlowWM](https://arxiv.org/abs/2606.29059) | Temporal consistency; mode-coverage metrics; stochastic-vs-deterministic proof; negative results |
| [CoLA-Flow](https://arxiv.org/abs/2601.23087) | Flow matching speed; latent-space flow; jitter from non-recurrent latents |
| [E3Flow](https://www.alphaxiv.org/abs/2603.23227) | Rectified flow speedup |
| [LatentToM](https://arxiv.org/html/2505.09144v1) | Ego/consensus factorization; robustness framing |
| [Dreaming of Others](https://arxiv.org/html/2605.31361v1) | Environment/teammate latent factorization |
| [Async VLA survey](https://arxiv.org/html/2605.08168) | Real-time chunking method comparison |
| [FutureRTC](https://arxiv.org/html/2607.24008) | Anticipatory conditioning under delay |
| [LARY](https://doi.org/10.48550/arxiv.2604.11689) / [JEPA-WM study](https://www.alphaxiv.org/abs/2512.24497) / [V-JEPA 2.1](https://arxiv.org/html/2603.14482v3) | Encoder selection |
| [FSQ](https://doi.org/10.48550/arxiv.2309.15505) | Discrete-latent option |
