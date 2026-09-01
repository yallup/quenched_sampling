"""Mutation kernel: blackjax MALA, preconditioned by the score."""
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from blackjax.mcmc import mala

_mala_kernel = mala.build_kernel()


def _tiny(x: Array) -> Array:
    return jnp.finfo(jnp.asarray(x).dtype).tiny


def _log_metric(pattern: Array, log_g2: Array, weights: Array,
                ess_frac: Array) -> Array:
    """Diagonal log sd from per-walker score statistics, sd_j proportional to
    (E[g_j^2 / |g|^2])^{-1/2} / |g|, shrunk toward flat."""
    weights = weights / jnp.sum(weights)
    stat = jnp.sum(weights[:, None] * pattern, axis=0)
    flat = jnp.mean(stat)
    # noise counted per lineage, not per row: clones are not fresh samples
    M_eff = jnp.maximum(1.0, pattern.shape[0] * ess_frac)
    noise = jnp.mean(
        jnp.sum(weights[:, None] * (pattern - stat[None, :]) ** 2, axis=0)
    ) / M_eff
    shrink = jnp.clip(1.0 - noise / jnp.clip(jnp.var(stat), _tiny(stat)), 0.0, 1.0)
    stat = flat + shrink * (stat - flat)
    # median of |g|^2: its upper tail is the level-boundary divergence
    return -0.5 * (jnp.log(jnp.clip(stat, _tiny(stat))) + jnp.median(log_g2))


def build_mutation(logdensity_fn: Callable, n_steps: int) -> Callable:
    """Build n_steps of MALA per walker in the diagonal metric sd.

    Returns a jitted (key, particles, E, step_size, sd, weights, ess_frac) ->
    (positions, mean acceptance, observed log sd).
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
