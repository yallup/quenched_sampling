"""The soft level family rho_E(x) ∝ pi(x) (E - U(x))_+^nu, U = -log L."""
from typing import Callable

import jax.numpy as jnp
from jax import Array

from blackjax.smc.ess import log_ess
from blackjax.smc.solver import dichotomy


def log_phi(s: Array, nu: float) -> Array:
    """Compute nu log (s)_+, -inf outside the level set."""
    s = jnp.asarray(s)
    # clip floor from the dtype: a literal underflows to 0 in float32
    return jnp.where(
        s > 0, nu * jnp.log(jnp.clip(s, jnp.finfo(s.dtype).tiny)), -jnp.inf
    )


def level_logdensity(
    U_fn: Callable, log_prior: Callable, nu: float
) -> Callable[[Array, Array], Array]:
    """Build log rho_E(x) as a function of (x, E). The prior must be smooth."""

    def logdensity(x: Array, E: Array) -> Array:
        return log_prior(x) + log_phi(E - U_fn(x), nu)

    return logdensity


def level_logw(U: Array, E: Array, E_new: Array, nu: float) -> Array:
    """Incremental log weight nu log[(E' - U)_+ / (E - U)_+].

    Evaluated as nu log(1 - dE/s) via log1mexp so only intensive quantities
    enter; exactly -inf outside either level.
    """
    tiny = jnp.finfo(jnp.asarray(U).dtype).tiny
    s = E - U
    dE = E - E_new
    t = jnp.log(jnp.clip(dE, tiny)) - jnp.log(jnp.clip(s, tiny))
    log1mexp = jnp.where(t > -0.693147, jnp.log(-jnp.expm1(t)), jnp.log1p(-jnp.exp(t)))
    return jnp.where((s > 0) & (dE < s), nu * log1mexp, -jnp.inf)


def ess_fraction(log_weights: Array) -> Array:
    """ESS as a fraction of the population."""
    return jnp.exp(log_ess(log_weights)) / log_weights.shape[0]


def boundary(U: Array, E: Array) -> Array:
    """Lowest admissible level, a few ulp above min U at the scale of E."""
    u_min = jnp.min(U)
    scale = jnp.maximum(jnp.maximum(jnp.abs(u_min), jnp.abs(E)), 1.0)
    return u_min + 4 * jnp.finfo(U.dtype).eps * scale


def next_level(U: Array, E: Array, nu: float, target_ess: float,
               log_W: Array | None = None) -> Array:
    """Find E' < E holding the ESS ratio ESS(W w)/ESS(W) at target_ess,
    by bisection on the descent delta = E - E'."""
    base = jnp.log(U.shape[0]) if log_W is None else log_ess(log_W)
    carried = jnp.zeros_like(U) if log_W is None else log_W
    target_val = base + jnp.log(target_ess)

    def criterion(delta: Array) -> Array:
        ess = log_ess(carried + level_logw(U, E, E - delta, nu))
        # NaN (empty level set) would read as converged
        return jnp.where(jnp.isnan(ess), -jnp.inf, ess) - target_val

    return E - dichotomy(criterion, jnp.zeros_like(E), E - boundary(U, E))
