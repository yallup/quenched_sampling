"""The soft microcanonical level family.

    rho_E(x) ∝ pi(x) (E - U(x))_+^nu,     U = -log L,

is the x|E conditional of the exact augmentation p(x, E) ∝ pi(x)(E-U)_+^nu e^-E,
whose x-marginal is the posterior. With the level volume
Lambda_nu(E) = E_pi[(E-U)_+^nu] this gives, for any nu > -1,

    Z = (1/Gamma(nu+1)) int Lambda_nu(E) e^-E dE.

nu interpolates the two incumbents: nu -> 0 is nested sampling's hard
constraint, whose score vanishes identically; nu -> infinity is tempering, one
effective temperature rather than a band. The usable window is the interior.
The positive part is not optional -- any factor whose log is locally linear in E
leaves a tempered posterior, which fails at a first-order transition.
"""
from typing import Callable

import jax.numpy as jnp
from jax import Array

from blackjax.smc.ess import log_ess
from blackjax.smc.solver import dichotomy


def log_phi(s: Array, nu: float) -> Array:
    """nu log (s)_+, and -inf outside the level set."""
    s = jnp.asarray(s)
    # floor from the dtype, not a literal: 1e-300 underflows to 0 in float32,
    # and the -inf branch of a `where` is still differentiated -> NaN gradient.
    return jnp.where(
        s > 0, nu * jnp.log(jnp.clip(s, jnp.finfo(s.dtype).tiny)), -jnp.inf
    )


def level_logdensity(
    U_fn: Callable, log_prior: Callable, nu: float
) -> Callable[[Array, Array], Array]:
    """log rho_E(x), as a function of (x, E). Its score is informative
    throughout the level set, which is what the hard constraint is not.

    The prior must be smooth: a hard indicator puts back the boundary the soft
    level exists to remove.
    """

    def logdensity(x: Array, E: Array) -> Array:
        return log_prior(x) + log_phi(E - U_fn(x), nu)

    return logdensity


def level_logw(U: Array, E: Array, E_new: Array, nu: float) -> Array:
    """log r = nu log[(E' - U)_+ / (E - U)_+], whose mean under rho_E is
    G(E')/G(E).

    Formed as log1mexp, not as a difference of logs. With s = E - U and
    dE = E - E', the ratio is 1 - dE/s: two intensive quantities. Differencing
    the logs instead subtracts extensive numbers -- U grows like D while the
    gap does not -- and spends log10(|U|/s) of float32's seven digits twice
    over. The two branches of log(1 - e^t) keep precision both at the new
    boundary (t near 0) and deep inside (t very negative).

    Outside either level the weight is exactly zero, computed rather than
    subtracted: (-inf) - (-inf) is NaN, and one logsumexp later that is the
    whole evidence.
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
    """Lowest level still containing a walker, held clear of min U.

    The margin is scaled by |E| as well as |min U| because the search returns
    E - delta: when |E| >> |min U| the ulp of that subtraction is larger than a
    margin scaled by min U alone, the bracket end lands below min U, every
    weight is -inf and the ESS is NaN.
    """
    u_min = jnp.min(U)
    scale = jnp.maximum(jnp.maximum(jnp.abs(u_min), jnp.abs(E)), 1.0)
    return u_min + 4 * jnp.finfo(U.dtype).eps * scale


def next_level(U: Array, E: Array, nu: float, target_ess: float,
               log_W: Array | None = None) -> Array:
    """The next level E' < E whose incremental weights retain `target_ess`.

    The ladder's only schedule: fixing the per-level weight variance fixes the
    price of a level and the step sizes follow. Solved on the descent
    delta = E - E' with blackjax's tempering dichotomy, whose contract matches
    -- decreasing in delta, positive at delta = 0, and returning the bracket end
    when the whole interval already meets the target, which is the case where
    the level set is admissible right down to min U.
    """
    # With carried weights the population is not uniform, so holding the ESS of
    # the increment alone over-steps: it charges the same information per level
    # regardless of how degenerate the cloud already is. Hold the RATIO
    # ESS(W w)/ESS(W) instead, which is what the increment actually costs.
    # Without weights the two coincide, since ESS(W) = n.
    base = jnp.log(U.shape[0]) if log_W is None else log_ess(log_W)
    carried = jnp.zeros_like(U) if log_W is None else log_W
    target_val = base + jnp.log(target_ess)

    def criterion(delta: Array) -> Array:
        ess = log_ess(carried + level_logw(U, E, E - delta, nu))
        # an empty level set has zero ESS; left as NaN the solver reads it as
        # converged and returns delta = 0, and the ladder silently stops.
        return jnp.where(jnp.isnan(ess), -jnp.inf, ess) - target_val

    return E - dichotomy(criterion, jnp.zeros_like(E), E - boundary(U, E))
