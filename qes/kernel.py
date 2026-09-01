"""Mutation kernel: blackjax's MALA, preconditioned by the score.

The estimator asks for invariance, not equilibration, so the kernel is a free
choice and plain MALA suffices. The metric is not free. The cloud variance
measures ensemble spread, which on a multimodal target is the mode separation,
and its between-mode term is maximised at balance -- so it rewards ensemble
collapse and measurably causes it. Read the metric from the score instead,

    sd_j = (E[g_j^2 / |g|^2])^{-1/2} / |g|,     g = grad log rho_E,

which is local, positive by construction, free (MALA computes g at every step),
and estimates the Hessian diagonal by the information identity
E[g g^T] = -E[grad^2 log rho_E].
"""
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from blackjax.mcmc import hmc, mala

_mala_kernel = mala.build_kernel()
_hmc_kernel = hmc.build_kernel()


def _tiny(x: Array) -> Array:
    return jnp.finfo(jnp.asarray(x).dtype).tiny


def _log_metric(pattern: Array, log_g2: Array, weights: Array,
                ess_frac: Array) -> Array:
    """log sd from the per-walker score statistics.

    The ensemble is a WEIGHTED atomic measure whose rows are not independent:
    branch-the-dead duplicates donors, and copies stay correlated through the
    mutation that follows. So the mean is taken under the carried weights, and
    its sampling noise is divided by the effective number of LINEAGES rather
    than the number of array rows. Counting clones as fresh samples
    under-shrinks a metric that should be flat, which is worst exactly where
    branching is heaviest.
    """
    weights = weights / jnp.sum(weights)
    stat = jnp.sum(weights[:, None] * pattern, axis=0)
    flat = jnp.mean(stat)  # = 1/D: the pattern sums to one
    M_eff = jnp.maximum(1.0, pattern.shape[0] * ess_frac)
    noise = jnp.mean(
        jnp.sum(weights[:, None] * (pattern - stat[None, :]) ** 2, axis=0)
    ) / M_eff
    shrink = jnp.clip(1.0 - noise / jnp.clip(jnp.var(stat), _tiny(stat)), 0.0, 1.0)
    stat = flat + shrink * (stat - flat)
    # scale from |g| too, so no part of the metric depends on positional spread;
    # median over walkers because |g| carries nu/(E-U), which diverges at the
    # level boundary.
    return -0.5 * (jnp.log(jnp.clip(stat, _tiny(stat))) + jnp.median(log_g2))


def build_mutation(logdensity_fn: Callable, n_steps: int) -> Callable:
    """n_steps of MALA per walker at level E in the diagonal metric `sd`.

    Returns (positions, mean acceptance, observed log sd). Walkers move in
    y = x/sd and are mapped back; the score is carried back to x likewise.
    Costs n_steps + 1 gradients per walker, the extra one initialising the MALA
    state, which cannot carry over because E and sd have both changed.
    """

    @jax.jit
    def mutate(key: Array, particles: Array, E: Array, step_size: Array,
               sd: Array, weights: Array, ess_frac: Array):
        def one_walker(key: Array, position: Array):
            logdensity = lambda y: logdensity_fn(y * sd, E)
            state = mala.init(position / sd, logdensity)

            def body(carry, key):
                state, pattern, log_g2 = carry
                state, info = _mala_kernel(key, state, logdensity, step_size)
                g = state.logdensity_grad / sd
                g2 = jnp.sum(g**2)
                # normalised per walker: nu/(E-U) multiplies the whole vector,
                # so dividing by |g|^2 removes the boundary divergence exactly
                # and leaves the pattern across coordinates.
                return (state, pattern + g**2 / g2, log_g2 + jnp.log(g2)), (
                    info.acceptance_rate
                )

            # accumulated in the carry: returning every step's gradient
            # materialises (N, n_steps, D) for a per-walker mean.
            init = (state, jnp.zeros_like(position), jnp.zeros((), position.dtype))
            (state, pattern, log_g2), acceptance = jax.lax.scan(
                body, init, jax.random.split(key, n_steps)
            )
            return (
                state.position * sd,
                pattern / n_steps,
                log_g2 / n_steps,
                jnp.mean(acceptance),
            )

        keys = jax.random.split(key, particles.shape[0])
        positions, pattern, log_g2, acceptance = jax.vmap(one_walker)(keys, particles)
        return positions, jnp.mean(acceptance), _log_metric(
            pattern, log_g2, weights, ess_frac)

    return mutate


def build_mutation_hmc(logdensity_fn: Callable, n_steps: int,
                       n_leapfrog: int) -> Callable:
    """As build_mutation, with HMC trajectories in place of MALA steps.

    The discriminating kernel for the freezing question: MALA's overdamped
    moves follow the energetically smoothest pathway at a structural
    transition, where a long leapfrog trajectory carries inertia across
    intermediate saddles. Same score metric, same y = x/sd frame -- the mass
    is the identity there because sd already carries the geometry. Costs
    n_steps * n_leapfrog gradients per walker per level.
    """

    @jax.jit
    def mutate(key: Array, particles: Array, E: Array, step_size: Array,
               sd: Array, weights: Array, ess_frac: Array):
        def one_walker(key: Array, position: Array):
            logdensity = lambda y: logdensity_fn(y * sd, E)
            state = hmc.init(position / sd, logdensity)
            imm = jnp.ones_like(position)

            def body(carry, key):
                state, pattern, log_g2 = carry
                # JITTERED step size, fixed L. Fixed-length HMC resonates on
                # near-harmonic targets -- and a crystallising cluster deep in
                # the ladder is one -- where a periodic orbit returns the
                # trajectory to its start and mixing dies. Randomising the
                # step over +-20% detunes every such orbit at no cost;
                # trajectory length stays static because the integrator's
                # scan requires it.
                key, k_jit = jax.random.split(key)
                eps = step_size * (0.8 + 0.4 * jax.random.uniform(k_jit))
                state, info = _hmc_kernel(
                    key, state, logdensity, eps, imm,
                    num_integration_steps=n_leapfrog,
                )
                g = state.logdensity_grad / sd
                g2 = jnp.sum(g**2)
                return (state, pattern + g**2 / g2, log_g2 + jnp.log(g2)), (
                    info.acceptance_rate
                )

            init = (state, jnp.zeros_like(position), jnp.zeros((), position.dtype))
            (state, pattern, log_g2), acceptance = jax.lax.scan(
                body, init, jax.random.split(key, n_steps)
            )
            return (
                state.position * sd,
                pattern / n_steps,
                log_g2 / n_steps,
                jnp.mean(acceptance),
            )

        keys = jax.random.split(key, particles.shape[0])
        positions, pattern, log_g2, acceptance = jax.vmap(one_walker)(keys, particles)
        return positions, jnp.mean(acceptance), _log_metric(
            pattern, log_g2, weights, ess_frac)

    return mutate
