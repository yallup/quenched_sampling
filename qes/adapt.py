"""Step-size and metric adaptation: constant-gain Robbins-Monro with Polyak
averaging for the descent, Nesterov dual averaging for the warmup."""
import math
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class Gain(NamedTuple):
    """gamma_k = rate * max((k+1)^-kappa, floor), plus the Polyak weight."""

    rate: float = 0.5
    kappa: float = 0.0  # constant gain: the optimum drifts with the level
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
    """Nesterov dual averaging on the log step size, as in Stan."""

    log_step: Array
    log_step_bar: Array
    h_bar: Array
    mu: Array
    count: Array


def dual_init(log_step, factor: float = 10.0) -> DualAveraging:
    log_step = jnp.asarray(log_step)
    zero = jnp.zeros_like(log_step)
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
