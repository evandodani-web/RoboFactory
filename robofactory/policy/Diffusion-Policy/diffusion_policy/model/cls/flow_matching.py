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

import torch

SIGMA_DISTRIBUTIONS = ("uniform", "logit_normal", "beta")
SOLVERS = ("euler", "midpoint")


class RectifiedFlowTransport:
    """Straight-line transport between Gaussian noise and clean action chunks.

    Stateless with respect to sampling: every call derives its own sigma schedule, so there
    is no step counter to desynchronize.
    """

    def __init__(
        self,
        sigma_dist: str = "uniform",
        sigma_dist_loc: float = 0.0,
        sigma_dist_scale: float = 1.0,
        shift: float = 1.0,
        timestep_scale: float = 1000.0,
        solver: str = "euler",
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
        if sigma_dist == "beta" and sigma_dist_scale <= 0:
            raise ValueError(
                f"beta needs a positive sigma_dist_scale, got {sigma_dist_scale}"
            )

        self.sigma_dist = sigma_dist
        self.sigma_dist_loc = sigma_dist_loc
        self.sigma_dist_scale = sigma_dist_scale
        self.shift = shift
        self.timestep_scale = timestep_scale
        self.solver = solver

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
            uniform = torch.rand(
                batch_size, device=device, dtype=dtype, generator=generator
            )
            sigma = uniform.pow(1.0 / self.sigma_dist_scale)

        return self.apply_shift(sigma)

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
                x = x + d_sigma * velocity
            else:
                # Midpoint: one extra model call buys second-order accuracy, which at very
                # low step counts can beat spending the same calls on more Euler steps.
                sigma_mid = sigma + 0.5 * d_sigma
                x_mid = x + 0.5 * d_sigma * velocity
                velocity_mid = model_fn(
                    x_mid, self.to_model_timestep(ones * sigma_mid)
                )
                x = x + d_sigma * velocity_mid

        return x
