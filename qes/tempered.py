"""Adaptive tempered SMC, matched to the quenched ladder in all but the path.

    p_beta(x) ∝ pi(x) L(x)^beta,   beta: 0 -> 1,

against the level family pi (E-U)_+^nu. Same population, same score-metric MALA
(`qes.kernel.build_mutation`, handed a different logdensity), same mutation
budget, same ESS criterion for the next rung, same warmup, and the same
Robbins-Monro step controller and gains. Matched by construction rather than by
assertion: an earlier version of this comparison preconditioned the tempered arm
by the cloud variance while the ladder used the score metric, which is exactly
the difference a multimodal target is built to expose.

Resampling is every stage here, unlike the ladder's branch-and-trigger cadence:
a tempered weight is strictly positive, so no particle ever leaves the support
and there is nothing to branch.

`n_beta` prescribes an equally spaced ladder instead of reading it off the
population. It is the rung-matched control -- adaptively, tempering reaches
beta = 1 in tens of stages, so "tempering misses the ordered phase" invites the
objection that it was starved of rungs. It is also the unbiased form: a fixed
schedule with unbiased resampling gives E[Zhat] = Z exactly, where an adaptive
one is only consistent.
"""
import math
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import logsumexp

from blackjax.smc.ess import ess_solver, log_ess
from blackjax.smc.resampling import systematic
from blackjax.smc.solver import dichotomy

from .adapt import Drift, Gain, drift_init, drift_update, dual_init, dual_update
from .kernel import build_mutation


def tempered_logdensity(U_fn: Callable, log_prior: Callable) -> Callable:
    """log p_beta(x) = log pi(x) - beta U(x), as a function of (x, beta)."""

    def logdensity(x: Array, beta: Array) -> Array:
        return log_prior(x) - beta * U_fn(x)

    return logdensity


class TemperedState(NamedTuple):
    key: Array
    particles: Array
    U: Array
    beta: Array
    log_Z: Array
    min_U: Array
    step: Drift
    metric: Drift


class StageInfo(NamedTuple):
    beta: Array
    log_Z: Array
    min_U: Array
    acceptance: Array
    ess: Array
    step_size: Array


class TemperedResult(NamedTuple):
    log_Z: float
    betas: np.ndarray
    log_Zs: np.ndarray
    snapshots: np.ndarray  # (n_stages, n_keep, D)
    log_weights: np.ndarray  # (n_stages, n_walkers), posterior weight per particle
    ess: np.ndarray
    step_sizes: np.ndarray
    acceptances: np.ndarray
    acceptance: float
    min_U: float
    n_stages: int
    pooled_kish: float
    grad_evals: float
    degenerate: bool


def next_beta(U: Array, beta: Array, target_ess: float) -> Array:
    """The next inverse temperature whose increment retains `target_ess`.

    blackjax's own tempering solver, on the incremental weight delta * log L
    with log L = -U. Its dichotomy returns the bracket end when the whole
    remaining range already meets the target, which is the jump straight to
    beta = 1.
    """
    delta = ess_solver(
        lambda particles: -particles,  # log L at the particles' energies
        U,
        target_ess,
        1.0 - beta,
        dichotomy,
    )
    return jnp.clip(beta + delta, beta, 1.0)


def _build_stage(U_fn, log_prior, n_steps, acc_target, step_gain, metric_gain,
                 n_keep, metric_mode="score"):
    mutate = build_mutation(tempered_logdensity(U_fn, log_prior), n_steps)
    U_batched = jax.vmap(U_fn)

    def mutate_and_adapt(state: TemperedState, ess_frac: Array):
        key, key_mutate = jax.random.split(state.key)
        # The cloud entering the mutation was just resampled, so its rows are
        # duplicated copies, not independent draws. The metric's shrinkage is
        # told how many lineages there really are -- the ESS of the incremental
        # weights that drove that resample -- exactly as the ladder is. The
        # weights themselves are uniform here because tempering resets them
        # every stage; the ladder carries them, and that is the only difference.
        n_w = state.particles.shape[0]
        x, acceptance, log_sd = mutate(
            key_mutate,
            state.particles,
            state.beta,
            jnp.exp(state.step.average),
            jnp.exp(state.metric.value),
            jnp.full((n_w,), 1.0 / n_w),
            ess_frac,
        )
        U = U_batched(x)
        state = state._replace(
            key=key,
            particles=x,
            U=U,
            min_U=jnp.minimum(
                state.min_U, jnp.min(jnp.where(jnp.isfinite(U), U, jnp.inf))
            ),
            step=drift_update(state.step, acceptance - acc_target, step_gain),
            # matched to the ladder: "frozen" warms at the first beta and
            # then holds, "unit" never has a matrix, and either way the step
            # size is the only thing adapting along the path.
            metric=(state.metric if metric_mode != "score" else drift_update(
                state.metric, log_sd - state.metric.value, metric_gain)),
        )
        return state, acceptance

    @jax.jit
    def stage(state: TemperedState, beta_new: Array):
        n = state.particles.shape[0]
        log_w = -(beta_new - state.beta) * state.U
        log_w = jnp.where(jnp.isfinite(log_w), log_w, -jnp.inf)
        log_Z = state.log_Z + logsumexp(log_w) - jnp.log(n)

        ess_frac = jnp.exp(log_ess(log_w)) / n
        key, key_resample = jax.random.split(state.key)
        idx = systematic(key_resample, jnp.exp(log_w - logsumexp(log_w)), n)
        state = state._replace(
            key=key,
            particles=state.particles[idx],
            U=state.U[idx],
            beta=beta_new,
            log_Z=log_Z,
        )
        state, acceptance = mutate_and_adapt(state, ess_frac)

        info = StageInfo(
            beta=beta_new,
            log_Z=log_Z,
            min_U=state.min_U,
            acceptance=acceptance,
            ess=ess_frac,
            step_size=jnp.exp(state.step.average),
        )
        # the posterior weight of every particle at this stage, log Z_beta
        # - (1-beta) U, which is what pools the whole path rather than its last
        # cloud -- the same accounting the ladder gets from its rung weights.
        return state, info, state.particles[:n_keep], log_Z - (1.0 - beta_new) * state.U

    @partial(jax.jit, static_argnames=("n",))
    def warmup(state: TemperedState, n: int):
        """Nesterov dual averaging at the first stage's beta, exactly as the
        ladder warms at E_0 -- the same recursion from `qes.adapt`, since the
        stage controller is constant-gain and proportional and so needs a start
        within a factor of a few. A step orders off scale gives acceptance
        0.000 and a frozen population that still returns a number."""

        def body(carry, key):
            st, da = carry
            n_w = st.particles.shape[0]
            x, acceptance, log_sd = mutate(
                key, st.particles, st.beta,
                jnp.exp(da.log_step), jnp.exp(st.metric.value),
                jnp.full((n_w,), 1.0 / n_w), jnp.asarray(1.0),
            )
            U = U_batched(x)
            st = st._replace(
                particles=x,
                U=U,
                min_U=jnp.minimum(
                    st.min_U, jnp.min(jnp.where(jnp.isfinite(U), U, jnp.inf))),
                metric=(st.metric if metric_mode == "unit" else drift_update(
                    st.metric, log_sd - st.metric.value, metric_gain)),
            )
            return (st, dual_update(da, acceptance, acc_target)), acceptance

        key, key_w = jax.random.split(state.key)
        carry = (state._replace(key=key), dual_init(state.step.value))
        (st, da), accs = jax.lax.scan(body, carry, jax.random.split(key_w, n))
        return st._replace(step=drift_init(da.log_step_bar)), accs

    return stage, warmup


def run(
    key: Array,
    U_fn: Callable,
    log_prior: Callable,
    sample_prior: Callable,
    *,
    n_walkers: int = 500,
    n_steps: int = 16,
    target_ess: float = 0.8,
    n_beta: int | None = None,
    n_warmup: int = 30,
    step_size: float = 0.1,
    acc_target: float = 0.574,
    step_gain: Gain = Gain(rate=0.5, kappa=0.0, floor=0.0, poly=0.15),
    metric_gain: Gain = Gain(rate=1.0, kappa=0.0, floor=0.0, poly=1.0),
    metric_mode: str = "score",
    max_stages: int = 20_000,
    n_keep: int = 40,
    verbose: int = 0,
) -> TemperedResult:
    """Estimate log Z by adaptive tempered SMC. Arguments mirror `qes.run`; only
    `n_beta` (a prescribed equally spaced ladder) is particular to this path."""
    key, key_init, key_run = jax.random.split(key, 3)
    x = sample_prior(key_init, n_walkers)
    U = jax.vmap(U_fn)(x)

    stage, warmup = _build_stage(
        U_fn, log_prior, n_steps, acc_target, step_gain, metric_gain, n_keep,
        metric_mode,
    )
    grid = None if n_beta is None else np.linspace(0.0, 1.0, int(n_beta) + 1)[1:]

    if metric_mode not in ("score", "frozen", "unit"):
        raise ValueError(f"metric_mode={metric_mode!r} is not one of "
                         "'score', 'frozen', 'unit'")
    if metric_mode == "unit":
        sd0 = jnp.ones_like(x[0])
    else:
        sd0 = jnp.sqrt(jnp.clip(jnp.var(x, axis=0), jnp.finfo(x.dtype).tiny))
        sd0 = sd0 / jnp.exp(jnp.mean(jnp.log(sd0)))
    state = TemperedState(
        key=key_run,
        particles=x,
        U=U,
        beta=jnp.zeros((), x.dtype),
        log_Z=jnp.zeros((), x.dtype),
        min_U=jnp.min(jnp.where(jnp.isfinite(U), U, jnp.inf)),
        step=drift_init(jnp.log(jnp.asarray(step_size, x.dtype))),
        metric=drift_init(jnp.log(sd0)),
    )

    # warm up at the FIRST stage's beta, as the ladder warms up at E_0
    beta1 = (
        float(next_beta(state.U, state.beta, target_ess))
        if grid is None
        else float(grid[0])
    )
    state, warm_acc = warmup(state._replace(beta=jnp.asarray(beta1, x.dtype)),
                             n_warmup)
    state = state._replace(beta=jnp.zeros((), x.dtype))
    if verbose:
        print(
            f"  warmup: beta1 {beta1:.4g}  acc {float(warm_acc[-1]):.3f}  "
            f"step {float(jnp.exp(state.step.average)):.3e}",
            flush=True,
        )

    betas, log_Zs, ess, steps, accs = [], [], [], [], []
    snapshots, log_weights = [], []
    for k in range(max_stages):
        if grid is None:
            beta_new = next_beta(state.U, state.beta, target_ess)
        else:
            # indexed by stage, not searched by value: the grid is float64 and
            # beta is the sampler's dtype, so a value search stalls the moment
            # the round trip lands below the rung it just took.
            beta_new = jnp.asarray(grid[min(k, len(grid) - 1)], x.dtype)

        state, info, snapshot, log_q = stage(state, beta_new)
        info = jax.device_get(info)

        betas.append(float(info.beta))
        log_Zs.append(float(info.log_Z))
        ess.append(float(info.ess))
        steps.append(float(info.step_size))
        accs.append(float(info.acceptance))
        snapshots.append(np.asarray(snapshot))
        log_weights.append(np.asarray(log_q))

        if verbose and k % verbose == 0:
            print(
                f"  stage {k:5d}  beta={info.beta:.6f}  logZ={info.log_Z:11.4f}  "
                f"acc={info.acceptance:.3f}  ess={info.ess:.3f}  "
                f"step={info.step_size:.2e}",
                flush=True,
            )
        if float(info.beta) >= 1.0:
            break

    dtype = x.dtype
    log_weights = np.asarray(log_weights, dtype)
    acceptance = float(np.mean(accs)) if accs else 0.0
    degenerate = acceptance < 0.05 or float(state.beta) < 1.0
    if degenerate:
        print(
            f"  WARNING: log Z is NOT trustworthy -- mean acceptance "
            f"{acceptance:.3f} over {len(betas)} stages, final beta "
            f"{float(state.beta):.4g}.",
            flush=True,
        )

    q = np.exp(log_weights - log_weights.max())
    return TemperedResult(
        log_Z=float(log_Zs[-1]) if log_Zs else float("nan"),
        betas=np.asarray(betas, dtype),
        log_Zs=np.asarray(log_Zs, dtype),
        snapshots=np.asarray(snapshots) if snapshots else np.empty((0, 0, 0)),
        log_weights=log_weights,
        ess=np.asarray(ess, dtype),
        step_sizes=np.asarray(steps, dtype),
        acceptances=np.asarray(accs, dtype),
        acceptance=acceptance,
        min_U=float(state.min_U),
        n_stages=len(betas),
        pooled_kish=float(q.sum() ** 2 / np.sum(q**2)) if q.size else 0.0,
        grad_evals=float((len(betas) + n_warmup) * n_walkers * (n_steps + 1)),
        degenerate=bool(degenerate),
    )


def posterior_sample(key: Array, result: TemperedResult, n: int) -> np.ndarray:
    """Equally weighted posterior draws pooled over every stage, weighting a
    particle at beta_k by log Z_beta - (1-beta_k) U. Scoring this arm on its
    final cloud alone while the ladder pools its whole path would not be the
    same question."""
    if result.snapshots.size == 0:
        raise ValueError("no snapshots were stored; run with n_keep > 0")
    n_keep = result.snapshots.shape[1]
    log_q = result.log_weights[:, :n_keep]
    q = np.exp(log_q - log_q.max()).ravel()
    q /= q.sum()
    draws = np.asarray(jax.random.choice(key, q.size, (n,), p=jnp.asarray(q)))
    return result.snapshots.reshape(-1, result.snapshots.shape[-1])[draws]
