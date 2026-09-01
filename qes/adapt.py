"""Robbins-Monro control with Polyak averaging, for both adapted quantities.

    theta_k+1    = theta_k    + gamma_k * signal,      gamma_k = rate * (k+1)^-kappa
    thetabar_k+1 = (1 - poly) thetabar_k + poly theta_k+1.

The gain is CONSTANT by default (kappa = 0): textbook Robbins-Monro decays it to
converge on a FIXED optimum, and ours moves at every level, so a decaying gain
stops tracking. Constant gain trades convergence for a bounded stationary
tracking error, which is the right trade on a quenched path. `floor` bounds the
decay if a kappa > 0 is asked for, so the recursion never stops tracking.

`poly` is an independent averaging weight, not the gain: tying the two makes the
average degenerate exactly when the gain is large enough to be useful (at gamma
= 1 the "average" is just the latest iterate). Averaging is what makes a gain
large enough to track also safe to use.

Which readout to use depends on the signal. An error signal that is only zero at
the optimum (acc - acc*) leaves the iterate a noise-driven walk: use `average`.
An observation of the quantity itself (obs - theta) already makes the iterate an
average: use `value`, since averaging it again is a second-order lag, measured
here as an acceptance dip while the level descends.
"""
import math
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class Gain(NamedTuple):
    """gamma_k = rate * max((k+1)^-kappa, floor), and the Polyak weight."""

    rate: float = 0.5
    kappa: float = 0.0  # 0 is constant gain: the optimum drifts with the level
    floor: float = 0.0
    poly: float = 0.15


class Drift(NamedTuple):
    value: Array
    average: Array
    count: Array


def drift_init(value) -> Drift:
    value = jnp.asarray(value)
    return Drift(value, value, jnp.zeros((), value.dtype))


def drift_update(state: Drift, signal: Array, gain: Gain) -> Drift:
    gamma = gain.rate * jnp.maximum((state.count + 1.0) ** -gain.kappa, gain.floor)
    value = state.value + gamma * signal
    average = (1.0 - gain.poly) * state.average + gain.poly * value
    return Drift(value, average, state.count + 1.0)


class DualAveraging(NamedTuple):
    """Nesterov dual averaging, as in Stan/NUTS. For the WARMUP only.

    The recursion above tracks a drifting optimum but only proportionally, so
    the value it starts from has to be within a factor of a few. A fixed step
    size is not: four orders out is routine across targets, and the failure is
    silent rather than slow -- at zero acceptance the particles are frozen, min
    U never improves, the dissection drives E onto that floor and the run exits
    on gap collapse with an evidence that looks converged.

    Dual averaging converges ON the target rate rather than bracketing it, and
    its averaged iterate is insensitive to the noise in any single acceptance.
    It is the right tool at a FIXED level and the wrong one once the ladder
    moves, which is why it warms up and then hands over.
    """

    log_step: Array
    log_step_bar: Array
    h_bar: Array
    mu: Array
    count: Array


def dual_init(log_step, factor: float = 10.0) -> DualAveraging:
    """From a LOG step, as both callers hold one."""
    log_step = jnp.asarray(log_step)
    zero = jnp.zeros_like(log_step)
    # mu is the point the iterate is shrunk toward: above the initial guess,
    # since a step that is too small is the failure that hides.
    return DualAveraging(log_step, log_step, zero, log_step + math.log(factor), zero)


def dual_update(
    state: DualAveraging,
    acceptance: Array,
    acc_target: float = 0.574,
    gamma: float = 0.05,
    t0: float = 10.0,
    kappa: float = 0.75,
) -> DualAveraging:
    m = state.count + 1.0
    eta = 1.0 / (m + t0)
    h_bar = (1.0 - eta) * state.h_bar + eta * (acc_target - acceptance)
    log_step = state.mu - jnp.sqrt(m) / gamma * h_bar
    w = m**-kappa
    log_step_bar = w * log_step + (1.0 - w) * state.log_step_bar
    return DualAveraging(log_step, log_step_bar, h_bar, state.mu, m)
