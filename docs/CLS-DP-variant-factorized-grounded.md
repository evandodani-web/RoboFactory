# CLS-DP Variant: Factorized Latent ("Study FG")

Design notes for the next CLS-DP variant. Builds on Study DET (deterministic latent), which
in turn holds everything else fixed against Study B — the run that reproduced the paper at
**61.0%** on LiftBarrier.

Related: [CLS-DP-replication-spec.md](CLS-DP-replication-spec.md) (paper extraction),
[CLS-DP-implementation-notes.md](CLS-DP-implementation-notes.md) (baseline + study log),
[CLS-DP-improvements.md](CLS-DP-improvements.md) (the full ranked idea list).

Status: **design only, nothing implemented.**

> This doc went through several rounds. Section 5 records the ideas that were considered and
> dropped, and why — the reasoning is more useful than the conclusions, because two of the
> dead ends died for reasons specific to CLS-DP's architecture that would not be obvious from
> reading PSG-JEPA.

---

## 1. The idea in one paragraph

CLS-DP compresses everything an agent needs to know about coordination into a single 256-d
latent, supervised by one objective: reconstruct every agent's future joint trajectory. This
variant **splits** that latent into a self half and a teammate half, each with its own
decoder, so the two jobs stop competing for the same capacity. It also adds a **diagnostic**
that the baseline is missing: measuring whether the prior's latent, *on its own*, is
sufficient — which is what deployment actually depends on.

Carried over from Study DET: the latent is deterministic, and the KL is replaced by its
unit-variance limit `0.5·‖z_E‖²`.

---

## 2. Before and after

**Today (Study B / DET):**

```
prior(image, text) ────────────────► z              (256)
privileged_enc(all agents' futures) ► residual      (256)

z_final = z + residual
decoder(own_state_now, z_final) ────► ALL agents' futures   (N, 8, 8)

loss = recon + beta·‖residual‖²
```

**Proposed (Study FG):**

```
prior(image, text) ────────────────► z_self (128) , z_team (128)
privileged_enc(all agents' futures) ► residual    (128)   ← z_team only

decoder_self(own_state_now, z_self)           ► MY future      (1,   8, 8)
decoder_team(own_state_now, z_team+residual)  ► THEIR futures  (N-1, 8, 8)

loss = recon_self + recon_team + beta·‖residual‖²

metrics only (no gradient):
  decoder_team(own_state_now, z_team)         ► prior-only reconstruction
  probe(frozen z_self) ─────────────────────► teammate state, should FAIL
```

Three loss terms, same count as today, **no new weights to tune**. Unchanged: deterministic
`z`, the Stage 2 action expert, the data pipeline, the SigLIP cache, `beta` and its warm-up,
Adam, 100 epochs. Latent total stays **256**, so Stage 2's cross-attention interface is
byte-identical to Study B.

---

## 3. Change 1: factorize the latent

`z → [z_self (128) ; z_team (128)]`, with separate decoders and the privileged residual on
`z_team` only.

**Why split.** One vector currently does two unrelated jobs: tracking my own task progress
and anticipating teammates. The paper half-admits this — Fig. 5 notes the ablated baseline
also fails a subtask "requiring no inter-agent coordination," and concludes the latent
encodes "the agent's own task progression as well as the dynamics of other agents."

**Why two decoders.** This is what makes the split real rather than nominal. With one decoder
over all agents' futures, nothing stops `z_self` from absorbing teammate information — the
names would be the only thing separating them. Two decoders confine each half
architecturally.

**Why the residual only on `z_team`.** The privileged signal is about coordination. My own
future is largely predictable from my own camera plus my own state and does not need
distilling. Concentrating alignment pressure on `z_team` spends `beta` where it matters.

The MA-kinematics encoder still takes **all** agents' futures as input — coordination is
relational, so it needs my trajectory to interpret theirs — but emits a residual only on
`z_team`.

---

## 4. Change 2: measure whether the prior alone is sufficient

This is diagnostics, not a loss term, and it may be the more valuable half of the variant.

**The gap.** During Stage 1 the decoder is *never* run on the prior's latent alone:

```
reconstruction = self.ma_decoder(own_state, latent)     # latent = mu_prior + mu_residual
```

Every reconstruction uses the **combined** latent. But at Stage 2 and at deployment the
residual does not exist — the action expert receives `z_prior` by itself. That is a
train/deploy mismatch of the same shape as exposure bias in sequence models: you train
conditioned on something that will not be there at test time.

**Why it matters more than it sounds.** The Stage 1 gate inherits the problem. `_diagnostics`
is computed from that same `squared_error`, so `ctx_recon_others` measures how well the
*combined* latent reconstructs teammates. **The gate can pass while the prior alone is
useless.** We have been treating that gate as authoritative.

**What to add:** a second decoder pass on `z_team` alone, logged as a metric with no
gradient. One extra decoder call (~1.7M params, negligible), no new weight, no risk. It
answers a question we currently cannot answer:

- Small gap between combined and prior-only → distillation is working as advertised.
- Large gap → the residual is carrying the load and the prior never learned the job.

**Only add the loss term if the gap is large.** Training on prior-only reconstruction is the
obvious fix, but it trades against what the residual is actually for (see section 5.4), so it
should be justified by a measurement rather than a guess.

**Also report both numbers in the gate**, not just the combined one.

---

## 5. Considered and dropped

### 5.1 `head_self`: ground `z_self` in the agent's own current state

Dropped. Redundant three ways over: the action expert already receives proprioception through
FiLM, the Stage 1 decoder receives `s_t^i` as an explicit input, and the reconstruction target
starts at `t+1` which is a hair away from `t` at this control rate. A loss term with weak
justification is not worth the tuning surface.

### 5.2 `head_team`: ground `z_team` in teammates' current states

This survived several rounds before dying. It looked like the strongest piece — teammates'
current pose is the one quantity the agent cannot obtain any other way, since it must read the
other arms off its own camera. The paper *claims* this happens (its attribution maps show
attention on teammates' joints) but nothing in the loss enforces it.

Dropped for two reasons:

1. The reconstruction target is `s^{1:N}_{t+1:t+H}`. These are consecutive control steps, so
   `q^j_{t+1}` and `q^j_t` differ by a tiny increment. Predicting one accurately means
   essentially knowing the other.
2. More damning: teammates' **current** pose is the *easiest* thing in the frame — it is
   literally visible. Their **future** is the hard part. So the head would spend a loss term
   and a tuning weight on the sub-problem reconstruction was always going to solve.

**Repurposed as a probe.** Train it on a **frozen** `z_self` and check whether teammate state
is recoverable. If the probe fails, the factorization worked. If it succeeds, the halves
merged despite the separate decoders and the split is cosmetic. That turns open question #4 in
the improvements doc from unanswerable into a number, and it mirrors what PSG-JEPA does for
measurement — their Table 1 is linear and MLP probes, kept separate from their objectives.

### 5.3 `head_delta`: PSG-JEPA transition grounding on a latent pair

The longest detour, and worth recording properly because the reasoning is subtle.

**What PSG-JEPA does.** A head reads a *pair* of latents `(z_t, z_{t+k})` and predicts the
joint displacement between them, `Δq = q_{t+k} − q_t`, for all horizons `k`. Displacement
rather than the intervening actions, because many action sequences can connect the same two
configurations while the displacement is fixed by the endpoints. Removing it was their single
largest planning hit, 95.0 → 81.3.

**First objection, wrong:** "the target is decomposable — the head can decode each latent and
subtract, so it teaches nothing." This is a real observation but the wrong conclusion.
Decomposability means no new *information*; it does not mean no new *gradient signal*.

Static grounding minimizes **absolute** error at each endpoint and says nothing about whether
those errors are *correlated* between nearby latents. Concretely, with `q_t = 1.000` and
`q_{t+1} = 1.010`, two decodes each carrying 1% absolute error in opposite directions give
`Δq̂ = −0.010` — right magnitude, wrong sign. Both endpoints are individually excellent and
the displacement is useless. Minimizing absolute error does not minimize the error of small
differences, and displacements between adjacent timesteps are small by construction.

PSG-JEPA says exactly this: static grounding "supervises each endpoint independently and does
not directly optimize a pairwise readout of their change."

**The real reason it does not port.** The *mechanism* transfers; the *pair* does not.

```
PSG-JEPA:   q_t     comes from z_t
            q_{t+k} comes from z_{t+k}       → the pair is unavoidable

CLS-DP:     q_t and q_{t+k} both come from
            z_t's decoded window             → one latent, one decoder pass
```

Their latent is a per-frame encoding; ours is a whole-future summary. Since `decoder_team`
already emits `q_{t+1} … q_{t+8}` from a single latent, the same reweighting is obtainable by
differencing that output — no second prior pass, no extra activation memory.

**The one thing the pair still buys.** A cross-latent head would correlate errors across
*separate prior invocations* — `z_t` from one control cycle, `z_{t+k}` from the next. Single
-latent differencing cannot do that. It is not "the latents sit near each other" but "what you
decode from consecutive latents stays mutually consistent," which is a genuine angle on the
jitter concern (W2 in the improvements doc), since the action expert re-conditions on a fresh
`z` every cycle.

| | Buys | Cost |
|---|---|---|
| Delta on decoder output | accurate velocity profile within one predicted window | free, one line |
| Cross-latent pair head | consistency of decoded belief across control cycles | 2 prior passes, ~2.5 GB extra activations at batch 512 |

**Verdict:** both are real and they are different. Take neither yet. Study DET is untrained,
so there is no variant baseline to attribute anything against, and stacking an extra-weight,
2.5 GB change onto an untested variant is how you end up unable to explain the result. If DET
rollouts look jittery, the cross-latent head becomes a well-motivated next move with a
specific symptom behind it.

### 5.4 Prior-only reconstruction as a loss (not just a metric)

Deferred, not rejected. See section 4 — measure the gap first.

The reason for caution: the residual is doing more than distillation. It absorbs the part of
the future that genuinely is not predictable from one camera frame, so the prior is not
punished for irreducible uncertainty. Remove that cushion and the prior is forced to fit noise,
and the classic response is mode-averaging. Training on prior-only reconstruction *alone*
would reduce to a plain predictor with no CVAE — a legitimate baseline, but it throws the
cushion away. A hybrid (both paths, weighted, possibly annealed from combined toward
prior-only, i.e. scheduled sampling) is the right shape if the measurement says it is needed.

---

## 6. What is genuinely non-redundant, for later

The grounding targets above all failed because they re-supervise robot joint values, which
the reconstruction already covers. The quantity that is **not** in the loss anywhere is the
**manipulated object's pose** — yet that is what the task is actually about.

Grounding a latent in object state would be new signal rather than a reweighting. Cost: object
pose is not in the zarr today, so it needs a change to `parse_pkl_to_zarr_multi.py` and a
re-parse. Worth doing as its own variant.

---

## 7. Runs to compare

| Run | Latent | Notes |
|---|---|---|
| Study B | stochastic, monolithic | reproduced the paper, **61.0%** on LiftBarrier |
| Study DET | deterministic, monolithic | isolates stochasticity — built, **not yet trained** |
| Study FG | deterministic, factorized | this doc |

Checkpoint prefixes follow the existing convention with a distinct tag per variant
(`*_ctxfg_*` / `*_clsdpfg_*`), so `eval_cls_sweep.sh`'s `CKPT_PREFIX` argument selects them and
results directories stay separate.

---

## 8. Open questions and risks

1. **Does the split hold?** Two decoders make it structural rather than hoped-for, but probe
   it anyway (section 5.2). If teammate state is recoverable from a frozen `z_self`, the split
   is cosmetic.
2. **Is 128 enough for `z_team`?** Coordination is arguably the harder job. Make the ratio
   configurable; 64/192 is worth trying.
3. **Interaction with the deterministic latent.** Study DET removes the prior's ability to
   express uncertainty. If DET underperforms because of mode-averaging, the factorization
   actually offers a targeted fix that the monolithic version cannot: reintroduce variance on
   `z_team` only, where the genuine ambiguity lives, while keeping `z_self` deterministic.
4. **Sequencing.** Study DET's number is informative for how to build this. Step 1
   (factorization + diagnostics) is cheap enough to develop in parallel, but wait for DET
   before adding anything on top.
