"""The quenched ladder: anchor, level loop, evidence estimator.

Each level mutates at E, dissects for E' < E, reweights, branches dead
walkers, and fully resamples only when the lineage-grouped ESS collapses.
Terminates on nested sampling's criterion, dlogz below the accumulated
integral.
"""
import math
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import logsumexp

from blackjax.smc.ess import log_ess
from blackjax.smc.resampling import systematic
from blackjax.smc.solver import dichotomy

from .adapt import Drift, Gain, drift_init, drift_update, dual_init, dual_update
from .kernel import build_mutation
from .level import (
    boundary,
    ess_fraction,
    level_logdensity,
    level_logw,
    log_phi,
    next_level,
)


class LadderState(NamedTuple):
    key: Array
    particles: Array
    log_W: Array  # carried log weights, reset to uniform on a full resample
    ancestor: Array  # lineage since the last full resample; the trigger reads it
    U: Array
    E: Array
    log_lambda: Array
    log_integral: Array
    min_U: Array
    step: Drift
    metric: Drift


class LevelInfo(NamedTuple):
    E: Array
    log_lambda: Array
    log_integral: Array
    min_U: Array
    acceptance: Array
    ess: Array
    step_size: Array


class LevelSample(NamedTuple):
    """One rung's contribution to the pooled cloud: retained walkers and
    their carried within-rung log weights."""

    walkers: Array
    log_w: Array


class Result(NamedTuple):
    """`Es` and `log_lambdas` are the ladder itself: the measured level volume
    at every rung, not just the endpoint."""

    log_Z: float
    Es: np.ndarray
    log_lambdas: np.ndarray
    snapshots: np.ndarray  # (n_levels, n_keep, D), the cloud at each rung
    log_weights: np.ndarray  # (n_levels, n_walkers), per-walker within a rung
    ess: np.ndarray
    step_sizes: np.ndarray
    acceptances: np.ndarray
    acceptance: float
    min_U: float
    n_levels: int
    grad_evals: float
    U_evals: int
    log_top: float
    log_tail_bound: float
    degenerate: bool


# ---------------------------------------------------------------------------
# initialisation
# ---------------------------------------------------------------------------
def _log_lambda(U: Array, E: Array, nu: float) -> Array:
    """log Lambda_nu(E) from prior draws."""
    return logsumexp(log_phi(E - U, nu)) - jnp.log(U.shape[0])


@partial(jax.jit, static_argnames=("nu", "n_walkers"))
def _start_level(U: Array, nu: float, n_walkers: int) -> Array:
    """Deepest E at which the prior draws hold ESS = n_walkers."""
    target_val = jnp.log(n_walkers)
    ess_at = lambda E: log_ess(log_phi(E - U, nu))
    # inflate above max U until the draws clear the target, then bisect down
    gap = jax.lax.while_loop(
        lambda g: ess_at(jnp.max(U) + g) < target_val,
        lambda g: 2 * g,
        jnp.maximum(jnp.std(U), 1.0),
    )
    E_hi = jnp.max(U) + gap
    criterion = lambda d: jnp.nan_to_num(ess_at(E_hi - d), nan=-jnp.inf) - target_val
    return E_hi - dichotomy(criterion, 0.0, E_hi - boundary(U, E_hi))


def anchor(
    key: Array,
    U_fn: Callable,
    sample_prior: Callable,
    nu: float,
    n_walkers: int,
    n_init: int,
    n_start: int | None = None,
    span: float = 120.0,
    spacing: float = 0.5,
):
    """Place E_0 from n_init prior draws, measure log Lambda(E_0), and
    resample the working population; the same draws also estimate the part of
    the evidence integral above E_0."""
    key_draw, key_resample = jax.random.split(key)
    x = sample_prior(key_draw, n_init)
    U = jax.lax.map(U_fn, x, batch_size=min(n_init, 512))

    # n_start uses an order statistic instead, for priors with divergent cores
    E0 = (_start_level(U, nu, n_walkers) if n_start is None
          else jnp.sort(U)[min(int(n_start), U.shape[0]) - 1])
    log_lambda0 = _log_lambda(U, E0, nu)

    grid = E0 + jnp.arange(0.0, span, spacing)
    log_lam = jax.vmap(lambda E: _log_lambda(U, E, nu))(grid)
    log_top = log_integrate(np.asarray(grid)[::-1], np.asarray(log_lam - grid)[::-1])

    log_w = log_phi(E0 - U, nu)
    idx = systematic(key_resample, jnp.exp(log_w - logsumexp(log_w)), n_walkers)
    return x[idx], E0, log_lambda0, log_top


# ---------------------------------------------------------------------------
# the level step
# ---------------------------------------------------------------------------
def _build_level(
    U_fn, log_prior, nu, n_steps, target_ess, acc_target, step_gain, metric_gain,
    n_keep, resample_ess, metric_mode="score",
):
    """Compile a mutate-and-adapt pass and one full level transition."""
    mutate = build_mutation(level_logdensity(U_fn, log_prior, nu), n_steps)
    U_batched = jax.vmap(U_fn)

    def mutate_and_adapt(state: LadderState):
        key, key_mutate = jax.random.split(state.key)
        # metric statistics use the carried weights and the lineage ESS
        n_w = state.particles.shape[0]
        w = jnp.exp(state.log_W - logsumexp(state.log_W))
        w = jnp.where(jnp.isfinite(w), w, 0.0)
        w = w / jnp.maximum(jnp.sum(w), 1e-300)
        grouped = jnp.bincount(state.ancestor, weights=w, length=n_w)
        ess_frac = jnp.clip(
            1.0 / jnp.maximum(jnp.sum(grouped ** 2), 1e-30) / n_w, 0.0, 1.0)
        x, acceptance, log_sd = mutate(
            key_mutate,
            state.particles,
            state.E,
            jnp.exp(state.step.average),
            jnp.exp(state.metric.value),
            w,
            ess_frac,
        )
        U = U_batched(x)
        state = state._replace(
            key=key,
            particles=x,
            U=U,
            # map non-finite U to +inf: jnp.minimum is sticky in NaN
            min_U=jnp.minimum(state.min_U,
                              jnp.min(jnp.where(jnp.isfinite(U), U, jnp.inf))),
            step=drift_update(state.step, acceptance - acc_target, step_gain),
            metric=(state.metric if metric_mode == "unit"
                    else drift_update(
                        state.metric, log_sd - state.metric.value,
                        metric_gain)),
        )
        return state, acceptance

    @jax.jit
    def level(state: LadderState):
        state, acceptance = mutate_and_adapt(state)
        n = state.particles.shape[0]

        E_new = next_level(state.U, state.E, nu, target_ess, state.log_W)
        log_w = level_logw(state.U, state.E, E_new, nu)
        # NaN weights and NaN walkers are dead, not poison: map to -inf
        bad = jnp.isnan(log_w) | ~jnp.isfinite(state.U)
        log_W = jnp.where(bad, -jnp.inf, state.log_W + log_w)
        log_lambda = state.log_lambda + logsumexp(log_W) - logsumexp(state.log_W)

        # running integral for the termination criterion only
        dE = state.E - E_new
        ok = dE > 0
        term = jnp.where(
            ok,
            jnp.log(jnp.where(ok, 0.5 * dE, 1.0))
            + jnp.logaddexp(state.log_lambda - state.E, log_lambda - E_new),
            -jnp.inf,
        )

        key, key_branch, key_resample = jax.random.split(state.key, 3)
        log_W = log_W - logsumexp(log_W)
        w = jnp.exp(log_W)

        # branch the dead: refill zero-weight slots with survivors drawn by
        # weight (one stratum per dead slot), splitting the donor's weight
        dead = ~jnp.isfinite(log_W)
        n_dead = jnp.sum(dead)
        cdf = jnp.cumsum(w)
        u = jax.random.uniform(key_branch, dtype=w.dtype)
        position = (u + jnp.cumsum(dead) - 1) / jnp.maximum(n_dead, 1)
        drawn = jnp.clip(
            jnp.searchsorted(cdf, position * cdf[-1], side="right"), 0, n - 1
        )
        idx = jnp.where(dead, drawn, jnp.arange(n))
        counts = jnp.bincount(idx, length=n)
        w = w[idx] / jnp.maximum(counts[idx], 1)
        w = w / jnp.sum(w)
        anc = state.ancestor[idx]

        # full resample only when the lineage-grouped ESS collapses; the
        # ungrouped ESS inflates with every branch and would never fire
        grouped = jnp.bincount(anc, weights=w, length=n)
        degenerate = 1.0 / jnp.maximum(jnp.sum(grouped ** 2), 1e-30) < resample_ess * n

        full = systematic(key_resample, w, n)
        idx2 = jnp.where(degenerate, full, jnp.arange(n))
        new = state._replace(
            key=key,
            particles=state.particles[idx][idx2],
            log_W=jnp.where(degenerate, jnp.full((n,), -jnp.log(n)),
                            jnp.log(w)[idx2]),
            ancestor=jnp.where(degenerate, jnp.arange(n), anc[idx2]),
            U=state.U[idx][idx2],
            E=E_new,
            log_lambda=log_lambda,
            log_integral=jnp.logaddexp(state.log_integral, term),
        )
        info = LevelInfo(
            E=E_new,
            log_lambda=log_lambda,
            log_integral=new.log_integral,
            min_U=new.min_U,
            acceptance=acceptance,
            # the ESS ratio the schedule holds
            ess=jnp.exp(log_ess(log_W) - log_ess(state.log_W)),
            step_size=jnp.exp(new.step.average),
        )
        # the pre-resampling cloud is the sample from rho_E
        return new, info, LevelSample(
            walkers=state.particles[:n_keep],
            log_w=log_W,
        )

    @partial(jax.jit, static_argnames=("n",))
    def warmup(state: LadderState, n: int):
        """Dual-averaging warmup at E_0: sets the step size, warms the
        metric, and equilibrates the anchor cloud."""

        def body(carry, key):
            st, da = carry
            n_w = st.particles.shape[0]
            x, acceptance, log_sd = mutate(
                key, st.particles, st.E,
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

    return level, warmup


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def run(
    key: Array,
    U_fn: Callable,
    log_prior: Callable,
    sample_prior: Callable,
    *,
    nu: float = 2.0,
    n_walkers: int = 500,
    n_steps: int = 16,
    target_ess: float = 0.8,
    dlogz: float = -10.0,
    n_init: int | None = None,
    n_start: int | None = None,
    n_warmup: int = 30,
    metric_mode: str = "score",
    resample_ess: float = 0.5,
    max_levels: int = 20_000,
    n_keep: int = 40,
    verbose: int = 0,
) -> Result:
    """Estimate log Z by a quenched ladder of soft level ensembles.

    Parameters
    ----------
    U_fn, log_prior, sample_prior
        U = -log L and log pi at one point (log_prior smooth and normalised),
        and (key, n) -> (n, D) prior draws.
    nu
        Level softness.
    target_ess
        ESS ratio each level's increment must retain: the schedule.
    dlogz
        Termination depth; must exceed the latent heat of any transition.
    n_init
        Anchor draws, default 10 * n_walkers.
    n_start
        Place E_0 at the n_start-th smallest prior energy; for priors with a
        divergent core.
    metric_mode
        "score" adapts the diagonal metric down the ladder; "unit" is the
        identity throughout.
    resample_ess
        Full-resample trigger on the lineage-grouped ESS fraction.
    """
    if metric_mode not in ("score", "unit"):
        raise ValueError(f"metric_mode={metric_mode!r} is not 'score' or 'unit'")
    acc_target, step_size = 0.574, 0.1
    step_gain = Gain(rate=0.5, kappa=0.0, floor=0.0, poly=0.15)
    metric_gain = Gain(rate=1.0, kappa=0.0, floor=0.0, poly=1.0)
    n_init = 10 * n_walkers if n_init is None else n_init
    if n_init < n_walkers:
        raise ValueError(f"n_init={n_init} cannot furnish n_walkers={n_walkers}")

    key, key_anchor, key_ladder = jax.random.split(key, 3)
    x0, E0, log_lambda0, log_top = anchor(
        key_anchor, U_fn, sample_prior, nu, n_walkers, n_init, n_start
    )
    level, warmup = _build_level(
        U_fn, log_prior, nu, n_steps, target_ess, acc_target, step_gain,
        metric_gain, n_keep, resample_ess, metric_mode,
    )

    if metric_mode == "unit":
        sd0 = jnp.ones_like(x0[0])
    else:
        sd0 = jnp.sqrt(jnp.clip(jnp.var(x0, axis=0), jnp.finfo(x0.dtype).tiny))
        sd0 = sd0 / jnp.exp(jnp.mean(jnp.log(sd0)))
    U0 = jax.vmap(U_fn)(x0)
    state = LadderState(
        key=key_ladder,
        particles=x0,
        log_W=jnp.full((n_walkers,), -jnp.log(n_walkers), dtype=E0.dtype),
        ancestor=jnp.arange(n_walkers),
        U=U0,
        E=E0,
        log_lambda=log_lambda0,
        log_integral=jnp.asarray(-jnp.inf, E0.dtype),
        min_U=jnp.min(jnp.where(jnp.isfinite(U0), U0, jnp.inf)),
        step=drift_init(jnp.log(jnp.asarray(step_size, x0.dtype))),
        metric=drift_init(jnp.log(sd0)),
    )

    state, warm_acc = warmup(state, n_warmup)
    if verbose:
        print(
            f"  warmup: E0 {float(E0):.4f}  "
            f"acc {float(warm_acc[-1]) if len(warm_acc) else float('nan'):.3f}  "
            f"step {float(jnp.exp(state.step.average)):.3e}",
            flush=True,
        )

    Es, log_lambdas = [float(E0)], [float(log_lambda0)]
    ess, steps, accs, snapshots, log_ws = [], [], [], [], []
    E_prev = float(E0)
    for k in range(max_levels):
        state, info, sample = level(state)
        info = jax.device_get(info)  # one host transfer per level

        # level spacing below the resolution of E: converged
        if not (info.E < E_prev):
            break

        Es.append(float(info.E))
        log_lambdas.append(float(info.log_lambda))
        ess.append(float(info.ess))
        steps.append(float(info.step_size))
        accs.append(float(info.acceptance))
        snapshots.append(np.asarray(sample.walkers))
        log_ws.append(np.asarray(sample.log_w))
        E_prev = float(info.E)

        if verbose and k % verbose == 0:
            print(
                f"  level {k:5d}  E={info.E:12.4f}  logLam={info.log_lambda:11.4f}  "
                f"acc={info.acceptance:.3f}  ess={info.ess:.3f}  "
                f"step={info.step_size:.2e}",
                flush=True,
            )

        if float(info.log_lambda - info.min_U - info.log_integral) < dlogz:
            break
        if info.E - info.min_U < 1e-10 * (Es[0] - info.min_U):
            break

    # keep host records in the sampler's own precision
    dtype = x0.dtype
    Es = np.asarray(Es, dtype)
    log_lambdas = np.asarray(log_lambdas, dtype)
    log_Z = float(
        np.logaddexp(log_integrate(Es, log_lambdas - Es), log_top)
        - math.lgamma(nu + 1)
    )
    min_U = float(info.min_U)
    acceptance = float(np.mean(accs)) if accs else 0.0
    n_levels = len(Es) - 1

    # a frozen ladder still returns a converged-looking number: flag it
    degenerate = acceptance < 0.05 or n_levels < 10
    if degenerate:
        print(
            f"  WARNING: log Z is NOT trustworthy -- mean acceptance "
            f"{acceptance:.3f} over {n_levels} levels; the ladder terminated "
            f"without mixing (min U {min_U:.3f}).",
            flush=True,
        )

    return Result(
        log_Z=log_Z,
        Es=Es,
        log_lambdas=log_lambdas,
        snapshots=np.asarray(snapshots) if snapshots else np.empty((0, 0, 0)),
        log_weights=np.asarray(log_ws) if log_ws else np.empty((0, 0)),
        ess=np.asarray(ess, dtype),
        step_sizes=np.asarray(steps, dtype),
        acceptances=np.asarray(accs, dtype),
        acceptance=acceptance,
        min_U=min_U,
        n_levels=n_levels,
        grad_evals=float((n_levels + n_warmup) * n_walkers * (n_steps + 1)),
        U_evals=int(n_init),
        log_top=float(log_top - math.lgamma(nu + 1)),
        log_tail_bound=float(log_lambdas[-1] - min_U - math.lgamma(nu + 1)),
        degenerate=bool(degenerate),
    )


def posterior_sample(key: Array, result: Result, n: int) -> np.ndarray:
    """Draw equally weighted posterior samples from the pooled path, weighting
    walker j of rung k by Lambda(E_k) e^-E_k dE_k times its within-rung
    weight."""
    snapshots = result.snapshots
    if snapshots.size == 0:
        raise ValueError("no snapshots were stored; run with n_keep > 0")
    k, n_keep = snapshots.shape[:2]
    Es, log_lambdas = result.Es[:k], result.log_lambdas[:k]
    dE = np.abs(np.gradient(Es)) if k > 1 else np.ones_like(Es)
    log_b = log_lambdas - Es + np.log(np.clip(dE, np.finfo(dE.dtype).tiny, None))

    log_a = result.log_weights[:k, :n_keep]
    a = np.exp(log_a - log_a.max(axis=1, keepdims=True))
    a /= a.sum(axis=1, keepdims=True)
    q = np.exp(log_b - log_b.max())[:, None] * a
    q = (q / q.sum()).ravel()

    draws = np.asarray(jax.random.choice(key, q.size, (n,), p=jnp.asarray(q)))
    return snapshots.reshape(-1, snapshots.shape[-1])[draws]


# ---------------------------------------------------------------------------
# quadrature
# ---------------------------------------------------------------------------
def log_integrate(E, log_f) -> float:
    """log int f dE over a descending grid, exact where log f is linear in E,
    with the trapezoid limit on flat intervals. Non-positive intervals are
    dropped."""
    E = np.asarray(E)
    log_f = np.asarray(log_f)
    dE = E[:-1] - E[1:]
    keep = dE > 0
    if not keep.any():
        return -np.inf
    dE, a, b = dE[keep], log_f[:-1][keep], log_f[1:][keep]
    hi, lo = np.maximum(a, b), np.minimum(a, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        exact = hi + np.log(-np.expm1(lo - hi)) - np.log(np.abs(a - b) / dE)
    trapezoid = np.log(0.5 * dE) + np.logaddexp(a, b)
    flat = np.abs(a - b) < np.sqrt(np.finfo(E.dtype).eps)
    return _logsumexp(np.where(flat, trapezoid, exact))


def _logsumexp(x: np.ndarray) -> float:
    top = x.max(initial=-np.inf)
    if not np.isfinite(top):
        return float(top)
    return float(top + np.log(np.exp(x - top).sum()))
