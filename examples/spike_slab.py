"""Minimal working example: spike-and-slab, graded against the closed form.

pi = N(0, I_D) and L(x) = sum_j h_j exp(-|x|^2 / 2 sigma_j^2) with
sigma = (1, 0.1), the spike's height set so it carries 90% of the evidence at
every D while occupying an exponentially small fraction of the prior mass. With
tau_j^2 = sigma_j^2/(1+sigma_j^2) the phases are disjoint shells at
|x| = tau_j sqrt(D), the gap between them is empty and the depth is extensive,
min U ~ -1.96 D: a first-order transition, on which tempering loses exactly
log 10 because the energy jumps a band no temperature visits.

    Z         = sum_j h_j tau_j^D
    posterior = sum_j w_j N(0, tau_j^2 I),   w_j ∝ h_j tau_j^D

so evidence and phase weights are both exact, and the two failure modes -- wrong
evidence, or right evidence with the wrong mode balance -- are distinguishable.

    uv run python examples/spike_slab.py --D 10
    uv run python examples/spike_slab.py --D 50 --seeds 3

The prior is smooth by design: a flat box imposed as an indicator puts back the
boundary the soft level exists to remove.
"""
import argparse
import time
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp as jax_logsumexp

import qes


@dataclass(frozen=True)
class SpikeSlab:
    D: int
    U_fn: Callable
    log_prior: Callable
    sample_prior: Callable
    log_Z: float
    spike_share: float
    min_U: float
    shell_radii: tuple
    dlogz: float
    _log_weights: tuple
    _tau: tuple

    def spike_fraction(self, x) -> float:
        """Fraction of draws in the spike, by exact posterior responsibility."""
        r2 = np.sum(np.asarray(x) ** 2, axis=-1)[:, None]
        tau2 = np.asarray(self._tau) ** 2
        log_r = (
            np.asarray(self._log_weights)
            - 0.5 * self.D * np.log(2 * np.pi * tau2)
            - r2 / (2 * tau2)
        )
        return float(np.mean(np.argmax(log_r, axis=-1) == 1))


def spike_slab(D: int, sigmas=(1.0, 0.1), spike_share: float = 0.9) -> SpikeSlab:
    sigmas = np.asarray(sigmas)
    tau2 = sigmas**2 / (1.0 + sigmas**2)
    ratio = spike_share / (1.0 - spike_share)
    log_h = np.array([0.0, np.log(ratio) + (D / 2) * np.log(tau2[0] / tau2[1])])
    log_Z = float(_logsumexp(log_h + (D / 2) * np.log(tau2)))
    log_w = log_h + (D / 2) * np.log(tau2) - log_Z

    jax_log_h, jax_sigmas = jnp.asarray(log_h), jnp.asarray(sigmas)

    def U_fn(x):
        return -jax_logsumexp(jax_log_h - jnp.sum(x**2) / (2 * jax_sigmas**2))

    def log_prior(x):
        return -0.5 * jnp.sum(x**2) - (D / 2) * jnp.log(2 * jnp.pi)

    def sample_prior(key, n):
        return jax.random.normal(key, (n, D))

    return SpikeSlab(
        D=D,
        U_fn=U_fn,
        log_prior=log_prior,
        sample_prior=sample_prior,
        log_Z=log_Z,
        spike_share=float(np.exp(log_w[1])),
        min_U=float(-_logsumexp(log_h)),
        shell_radii=tuple(np.sqrt(tau2) * np.sqrt(D)),
        # scales with the transition, and has to: the coexistence gap is ~D nats
        # wide and carries almost no mass, so a fixed threshold halts in the
        # background phase before the spike is reached at all.
        dlogz=-2.5 * D - 20.0,
        _log_weights=tuple(log_w),
        _tau=tuple(np.sqrt(tau2)),
    )


def _logsumexp(x: np.ndarray) -> float:
    top = x.max()
    return float(top + np.log(np.exp(x - top).sum()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--D", type=int, default=10)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--walkers", type=int, default=500)
    p.add_argument("--steps", type=int, default=16, help="mutation steps per level")
    p.add_argument("--nu", type=float, default=2.0)
    p.add_argument("--target-ess", type=float, default=0.8)
    p.add_argument("--posterior-draws", type=int, default=20_000)
    p.add_argument("--verbose", type=int, default=0, help="print every k levels")
    p.add_argument("--method", choices=("qes", "tempered"), default="qes",
                   help="the quenched ladder, or the tempered SMC baseline "
                        "matched to it in population, kernel, budget and ESS")
    p.add_argument("--n-beta", type=int, default=None,
                   help="tempered only: prescribe an equally spaced ladder of "
                        "this many stages instead of reading it off the ESS")
    args = p.parse_args()

    target = spike_slab(args.D)
    print(
        f"spike-and-slab, D = {target.D}\n"
        f"  exact log Z      {target.log_Z:12.4f}\n"
        f"  exact min U      {target.min_U:12.4f}\n"
        f"  spike share      {target.spike_share:12.4f}\n"
        f"  phase shells at  |x| = {target.shell_radii[0]:.3f} (slab), "
        f"{target.shell_radii[1]:.3f} (spike)\n"
        f"  dlogz            {target.dlogz:12.1f}\n"
    )

    log_Zs = []
    for seed in range(args.seeds):
        key_run, key_post = jax.random.split(jax.random.key(seed))
        start = time.perf_counter()
        if args.method == "qes":
            result = qes.run(
                key_run,
                target.U_fn,
                target.log_prior,
                target.sample_prior,
                nu=args.nu,
                n_walkers=args.walkers,
                n_steps=args.steps,
                target_ess=args.target_ess,
                dlogz=target.dlogz,
                verbose=args.verbose,
            )
            draws = qes.posterior_sample(key_post, result, args.posterior_draws)
            rungs, rung_name = result.n_levels, "levels"
            extra = (f"  unvisited bound  "
                     f"{result.log_tail_bound - result.log_Z:12.1f} nats\n")
        else:
            result = qes.tempered.run(
                key_run,
                target.U_fn,
                target.log_prior,
                target.sample_prior,
                n_walkers=args.walkers,
                n_steps=args.steps,
                target_ess=args.target_ess,
                n_beta=args.n_beta,
                verbose=args.verbose,
            )
            draws = qes.tempered.posterior_sample(
                key_post, result, args.posterior_draws)
            rungs, rung_name = result.n_stages, "stages"
            extra = f"  pooled Kish      {result.pooled_kish:12.1f}\n"
        runtime = time.perf_counter() - start
        log_Zs.append(result.log_Z)
        print(
            f"seed {seed}\n"
            f"  log Z            {result.log_Z:12.4f}   "
            f"(bias {result.log_Z - target.log_Z:+.4f} nats)\n"
            f"  min U            {result.min_U:12.4f}   "
            f"(exact {target.min_U:.4f})\n"
            f"  spike fraction   {target.spike_fraction(draws):12.4f}   "
            f"(exact {target.spike_share:.4f})\n"
            f"  {rung_name:15s}{rungs:12d}\n"
            f"  gradients        {result.grad_evals:12.3g}\n"
            f"  acceptance       {result.acceptance:12.3f}   "
            f"(min {result.acceptances.min():.3f})\n"
            f"  mean ESS         {np.mean(result.ess):12.3f}\n"
            f"{extra}"
            f"  wall             {runtime:12.1f} s\n"
        )

    if args.seeds > 1:
        # never average log Z: the estimator is unbiased for Z, so averaging its
        # logarithm imports a Jensen penalty of half the variance.
        log_Zs = np.asarray(log_Zs)
        log_mean = log_Zs.max() + np.log(np.exp(log_Zs - log_Zs.max()).mean())
        print(
            f"over {args.seeds} seeds:  log mean Z {log_mean:.4f}  "
            f"(bias {log_mean - target.log_Z:+.4f}), spread {log_Zs.std():.4f}"
        )


if __name__ == "__main__":
    main()
