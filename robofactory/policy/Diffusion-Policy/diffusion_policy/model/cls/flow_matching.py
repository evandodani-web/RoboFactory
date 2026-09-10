"""Rectified-flow / conditional-OT transport for the CLS-DP Stage 2 action expert.

Replaces the paper's 100-step DDPM reverse chain with a straight-line ODE that samples in a
handful of Euler steps. The U-Net, its conditioning, and the z cross-attention are all
unchanged -- only what the network is asked to predict, and how that prediction is
integrated, differ.

Conventions follow diffusers' `FlowMatchEulerDiscreteScheduler` so that our arithmetic can
be cross-checked against a reference implementation (see verify_cls_dp.py). With `sigma`
running from 1 (pure noise) down to 0 (data):

    x_sigma = sigma * noise + (1 - sigma) * x1          forward interpolation
    v       = noise - x1                                velocity target, constant in sigma
    x <- x + (sigma_next - sigma) * v_hat               reverse Euler step

Two identities fall out of the above and are asserted in the verification suite:

    x_sigma = sigma * v + x1        =>      x1 = x_sigma - sigma * v

so a single Euler step from sigma=1 recovers exactly the model's implied clean prediction.
The sampler is written through that identity -- `x <- x1_hat + sigma_next * v` -- which is
algebraically the same Euler step but exposes the clean prediction, so `clamp_x1` can bound
it the way the DDPM head's `clip_sample=True` bounds its own x0 estimate. See `step_to`.

Why this is hand-rolled rather than delegated to diffusers, despite matching its
conventions -- all three verified against the pinned diffusers==0.32.2 source:

  * `scale_noise` cannot be used for training. It resolves sigma through
    `index_for_timestep`, an exact-equality lookup against a discrete 1000-entry grid, so a
    continuous sigma matches nothing and raises IndexError.
  * `FlowMatchEulerDiscreteScheduler.config` has no `prediction_type`, which the inherited
    DDPM `compute_loss` reads unconditionally.
  * `step` carries hidden mutable state (`_step_index`, reset only by `set_timesteps`),
    which is a footgun next to EMA and repeated sampling calls.

The reference implementation is therefore used as a test oracle, not as a dependency.
"""

from typing import Optional

import torch

SIGMA_DISTRIBUTIONS = ("uniform", "logit_normal", "beta")
SOLVERS = ("euler", "midpoint", "heun")

# `sigma_dist_scale` means something different per distribution -- `a` in Beta(a, 1) versus
# the std of the logit-normal -- so a single shared default would silently mis-tune whichever
# one it was not chosen for. Left as None by the caller, each resolves to its field standard:
# 1.5 is pi0/pi0.5/SmolVLA/GR00T's Beta(1.5, 1.0), 1.0 is SD3's logit-normal.
DEFAULT_SIGMA_DIST_SCALE = {"uniform": 1.0, "logit_normal": 1.0, "beta": 1.5}


class RectifiedFlowTransport:
    """Straight-line transport between Gaussian noise and clean action chunks.

    Stateless with respect to sampling: every call derives its own sigma schedule, so there
    is no step counter to desynchronize.
    """

    def __init__(
        self,
        sigma_dist: str = "beta",
        sigma_dist_loc: float = 0.0,
        sigma_dist_scale: Optional[float] = None,
        shift: float = 1.0,
        timestep_scale: float = 1000.0,
        solver: str = "euler",
        sigma_min: float = 1e-4,
        clamp_x1: Optional[float] = 1.0,
    ):
        if sigma_dist not in SIGMA_DISTRIBUTIONS:
            raise ValueError(
                f"sigma_dist must be one of {SIGMA_DISTRIBUTIONS}, got {sigma_dist!r}"
            )
        if solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
        if shift <= 0:
            raise ValueError(f"shift must be positive, got {shift}")
        if timestep_scale <= 0:
            raise ValueError(f"timestep_scale must be positive, got {timestep_scale}")
        if not 0.0 <= sigma_min < 1.0:
            raise ValueError(f"sigma_min must be in [0, 1), got {sigma_min}")
        if sigma_dist_scale is None:
            sigma_dist_scale = DEFAULT_SIGMA_DIST_SCALE[sigma_dist]
        if sigma_dist == "beta" and sigma_dist_scale <= 0:
            raise ValueError(
                f"beta needs a positive sigma_dist_scale, got {sigma_dist_scale}"
            )
        if clamp_x1 is not None and clamp_x1 <= 0:
            raise ValueError(f"clamp_x1 must be positive or None, got {clamp_x1}")

        self.sigma_dist = sigma_dist
        self.sigma_dist_loc = sigma_dist_loc
        self.sigma_dist_scale = sigma_dist_scale
        self.shift = shift
        self.timestep_scale = timestep_scale
        self.solver = solver
        self.sigma_min = sigma_min
        self.clamp_x1 = clamp_x1

    # ------------------------------------------------------------------ sigma

    def apply_shift(self, sigma):
        """SD3/Flux time shift. Identity at shift=1.0, which is the default here.

        CLS-DP-improvements.md section 4 argues the optimal shift for this task is near
        zero because the action tensor is only 8x8 = 64 values, so the knob exists to be
        swept, not to be turned on by default.
        """
        if self.shift == 1.0:
            return sigma
        return self.shift * sigma / (1.0 + (self.shift - 1.0) * sigma)

    def sample_sigma(self, batch_size, device, dtype=torch.float32, generator=None):
        """Draw the training noise levels, shape (batch_size,), in [0, 1]."""
        if self.sigma_dist == "uniform":
            sigma = torch.rand(
                batch_size, device=device, dtype=dtype, generator=generator
            )
        elif self.sigma_dist == "logit_normal":
            # SD3's choice: concentrates samples on mid-sigma, where the transport is
            # hardest to learn and the endpoints are least informative.
            normal = torch.randn(
                batch_size, device=device, dtype=dtype, generator=generator
            )
            sigma = torch.sigmoid(self.sigma_dist_loc + self.sigma_dist_scale * normal)
        else:
            # Beta(a, 1) has CDF x^a, so its inverse CDF is u^(1/a). Doing it by inverse
            # transform rather than torch.distributions keeps `generator` honoured, which
            # matters for the determinism checks. b is fixed at 1, which covers the
            # pi0-style "emphasise high noise" case (a > 1) that motivates this option.
            #
            # This is the default. a=1.5 reproduces the Beta(1.5, 1.0) that pi0, pi0.5,
            # SmolVLA, GR00T and WALL-X all converged on; sigma=1 is noise here exactly as
            # in openpi's `x_t = t * noise + (1 - t) * actions`, so the bias direction
            # carries over without a flip. It puts ~65% of training mass above sigma=0.5
            # against uniform's 50%, which is the half of the path a few-step solver
            # crosses in its first, largest, and least recoverable steps.
            uniform = torch.rand(
                batch_size, device=device, dtype=dtype, generator=generator
            )
            sigma = uniform.pow(1.0 / self.sigma_dist_scale)

        return self.apply_shift(sigma).clamp(min=self.sigma_min, max=1.0)

    def sigma_schedule(self, num_steps, device, dtype=torch.float32):
        """Descending sigmas for sampling, shape (num_steps + 1,), ending exactly at 0.

        Mirrors `FlowMatchEulerDiscreteScheduler.set_timesteps`: a linspace from 1 down to
        1/timestep_scale, shifted, with a terminal zero appended so the final step lands on
        the data manifold rather than near it.
        """
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        base = torch.linspace(
            1.0, 1.0 / self.timestep_scale, num_steps, device=device, dtype=dtype
        )
        sigmas = self.apply_shift(base)
        return torch.cat([sigmas, torch.zeros(1, device=device, dtype=dtype)])

    def to_model_timestep(self, sigma):
        """Scale sigma into the range the U-Net's sinusoidal embedding expects.

        This is not cosmetic. `SinusoidalPosEmb` builds frequencies spanning 1.0 down to
        1e-4, so a raw sigma in [0, 1] varies meaningfully in roughly one of 128 embedding
        dimensions and the network is effectively blind to its own noise level. Scaling to
        [0, 1000] is what diffusers feeds its own U-Nets.
        """
        return sigma * self.timestep_scale

    # -------------------------------------------------------------- forward process

    @staticmethod
    def _broadcast(sigma, like):
        """Reshape (B,) to (B, 1, ..., 1) so it broadcasts against a (B, T, C) tensor."""
        return sigma.reshape(sigma.shape[0], *([1] * (like.dim() - 1)))

    def interpolate(self, x1, noise, sigma):
        """x_sigma = sigma * noise + (1 - sigma) * x1."""
        s = self._broadcast(sigma, x1)
        return s * noise + (1.0 - s) * x1

    @staticmethod
    def velocity_target(x1, noise):
        """v = noise - x1, i.e. dx/dsigma along the straight path. Constant in sigma."""
        return noise - x1

    def implied_x1(self, x_sigma, sigma, velocity):
        """Invert the interpolation: x1 = x_sigma - sigma * v.

        Used by the `clean`-space temporal-consistency term and by the verification suite.
        """
        return x_sigma - self._broadcast(sigma, x_sigma) * velocity

    # -------------------------------------------------------------- reverse process

    def step_to(self, x, sigma, sigma_target, velocity):
        """Advance along the straight path from `sigma` to `sigma_target`.

        Algebraically this is the plain Euler step, since

            x1 + sigma_target * v = (x - sigma * v) + sigma_target * v
                                  = x + (sigma_target - sigma) * v

        but routing it through the implied clean sample is what lets `clamp_x1` bound the
        *clean prediction* rather than the noisy iterate, which is what DDPM's
        `clip_sample=True` does to its own x0 estimate. Two consequences worth knowing:

          * The schedule ends at sigma_target = 0, so the last step returns the clamped x1
            itself. A clamped sampler therefore cannot emit an out-of-range action.
          * No division by sigma is involved, so this stays well defined as sigma -> 0.

        With clamp_x1=None the arithmetic is bit-for-bit the previous implementation, which
        is what the diffusers cross-check in verify_cls_dp.py pins.
        """
        x1 = x - sigma * velocity
        if self.clamp_x1 is not None:
            x1 = x1.clamp(-self.clamp_x1, self.clamp_x1)
        return x1 + sigma_target * velocity

    def sample(
        self,
        model_fn,
        shape,
        num_steps,
        device,
        dtype=torch.float32,
        generator=None,
        noise=None,
    ):
        """Integrate the ODE from noise at sigma=1 to data at sigma=0.

        Args:
            model_fn: callable (x, t_model) -> velocity, where `t_model` is the already
                scaled timestep of shape (B,). The transport owns the scaling so callers
                cannot forget it.
            num_steps: Euler steps. Model calls are `num_steps` for the euler solver and
                `2 * num_steps` for midpoint.
            noise: optional starting point, for reproducibility in tests.

        Returns:
            The sample at sigma = 0.
        """
        if noise is None:
            x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        else:
            x = noise.to(device=device, dtype=dtype)

        sigmas = self.sigma_schedule(num_steps, device=device, dtype=dtype)
        ones = torch.ones(x.shape[0], device=device, dtype=dtype)

        for i in range(num_steps):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            d_sigma = sigma_next - sigma

            velocity = model_fn(x, self.to_model_timestep(ones * sigma))
            if self.solver == "euler":
                x = self.step_to(x, sigma, sigma_next, velocity)
            elif self.solver == "midpoint":
                # Midpoint: one extra model call buys second-order accuracy, which at very
                # low step counts can beat spending the same calls on more Euler steps.
                sigma_mid = sigma + 0.5 * d_sigma
                x_mid = self.step_to(x, sigma, sigma_mid, velocity)
                velocity_mid = model_fn(
                    x_mid, self.to_model_timestep(ones * sigma_mid)
                )
                x = self.step_to(x, sigma, sigma_next, velocity_mid)
            else:
                # Heun / improved Euler: the 2-NFE method used by EDM and k-diffusion.
                # Skip the correcting eval on the last step (sigma_next = 0); that step
                # is already tiny (1/timestep_scale -> 0) and t=0 is not a training point.
                x_euler = self.step_to(x, sigma, sigma_next, velocity)
                if i + 1 < num_steps:
                    velocity_next = model_fn(
                        x_euler, self.to_model_timestep(ones * sigma_next)
                    )
                    x = self.step_to(
                        x, sigma, sigma_next, 0.5 * (velocity + velocity_next)
                    )
                else:
                    x = x_euler

        return x
