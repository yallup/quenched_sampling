"""The sampler: initialisation, the level loop, the evidence estimator.

One iteration, from N walkers approximately distributed as rho_E:

    mutate at E and adapt -> dissect for E' < E -> reweight, accumulate
    log Lambda(E') = log Lambda(E) + log mean_i w_i, resample by w.

Unbiasedness needs invariance, not equilibration: this is the standard SMC
normalising-constant estimator, unbiased for any number of mutation steps
provided each kernel is invariant for its own level. The population never has
to have relaxed, so the level can be driven down as fast as the weights allow.
(Choosing E' from the population does cost exact unbiasedness at finite N, as
it does in nested sampling; the estimator stays consistent.)

Resampling is done at every level. Unlike tempering, an incremental weight here
can be exactly zero, so deferring it leaves dead walkers in the population.

Termination is nested sampling's: stop when Lambda(E) e^-min U, the optimistic
bound on the unvisited remainder, falls dlogz below the accumulated integral.
dlogz must exceed the latent heat of any transition -- at coexistence the
integrand goes flat and a tight criterion reports the plateau as the peak.
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
from .kernel import build_mutation, build_mutation_hmc
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
    """What each rung contributes to the pooled cloud: the retained walkers,
    their per-walker log weight, and any tracked summary of them.

    The weight is the CARRIED weight times this transition's increment,
    normalised, evaluated at the pre-mutation positions, so a walker is paired
    with the weight it actually carries. The increment alone is not enough once
    weights survive a level: between full resamples the ensemble is unequally
    weighted by construction. Assuming an equal share within a rung is wrong
    wherever branching has duplicated walkers, and that is exactly the case the
    pooled diagnostics are used to detect.
    """

    walkers: Array
    log_w: Array
    stat: Array


class Result(NamedTuple):
    """`Es` and `log_lambdas` are the ladder itself: the measured level volume
    at every rung, not just the endpoint."""

    log_Z: float
    Es: np.ndarray
    log_lambdas: np.ndarray
    snapshots: np.ndarray  # (n_levels, n_keep, D), the cloud at each rung
    stats: np.ndarray  # (n_levels, n_walkers, k), track(cloud) at each rung
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
    """The deepest E at which the prior draws still support n_walkers effective
    samples: the reweighting has to furnish N walkers, and log Lambda(E_0),
    which everything downstream telescopes off, is measured from the same draws.
    """
    target_val = jnp.log(n_walkers)
    ess_at = lambda E: log_ess(log_phi(E - U, nu))
    # inflate above max U until the draws clear the target, then bisect down.
    # every weight tends to the same value as the gap grows, so this terminates.
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
    """Place E_0 from n_init prior draws and reweight them to n_walkers.

    The same draws do three jobs: place E_0 at ESS = N, measure log Lambda(E_0),
    and supply the first walkers. They also pay for the part of the integral
    above the ladder -- Z integrates over all E, and for E > E_0 the level
    weights are better conditioned still, so Lambda is estimated directly there.
    """
    key_draw, key_resample = jax.random.split(key)
    x = sample_prior(key_draw, n_init)
    U = jax.lax.map(U_fn, x, batch_size=min(n_init, 512))

    # n_start places E_0 at an ORDER STATISTIC instead, nested sampling's
    # initialisation. Required whenever the prior has a divergent repulsive
    # core: _start_level searches downward from max(U) + gap, and for a hard
    # sphere or 12-6 core max(U) is set by the closest accidental pair, which
    # is 10^15 or worse. The search then begins that far above anything
    # physical and cannot recover -- measured on LJ38, E_0 came back as 10^9
    # or NaN at every container size, while the order statistic is untouched
    # by the tail because it only counts draws.
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
    n_keep, track, resample_ess, n_leapfrog, metric_mode="score",
):
    """Compile a mutate-and-adapt pass, and one level built on it.

    A level is one jitted function and one small record: the bisection in
    particular has to stay on device, or a device-to-host sync per iteration
    makes the schedule, not the kernel, the cost of the ladder.
    """
    mutate = (build_mutation(level_logdensity(U_fn, log_prior, nu), n_steps)
              if n_leapfrog is None else
              build_mutation_hmc(level_logdensity(U_fn, log_prior, nu),
                                 n_steps, n_leapfrog))
    U_batched = jax.vmap(U_fn)

    def mutate_and_adapt(state: LadderState):
        key, key_mutate = jax.random.split(state.key)
        # The metric is read from a WEIGHTED, branched ensemble. Supply the
        # carried weights so its mean matches the measure the walkers actually
        # represent, and the lineage-grouped ESS -- the same quantity the
        # resample trigger reads -- so its shrinkage counts independent
        # lineages rather than duplicated array rows.
        n_w = state.particles.shape[0]
        if metric_mode == "score_rows":
            # THE PRE-PATCH ESTIMATOR, recovered exactly rather than kept as a
            # second copy: with uniform weights the weighted mean collapses to
            # the plain mean, and with ess_frac 1 the shrinkage divides by the
            # row count. Both forms therefore run through one code path, which
            # is the only way the comparison between them means anything.
            w = jnp.full((n_w,), 1.0 / n_w, state.particles.dtype)
            ess_frac = jnp.asarray(1.0, state.particles.dtype)
        else:
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
            # A non-finite walker is about to be branched away and must not
            # poison the bound on its way out. nanmin is not enough: min_U is
            # carried with jnp.minimum, which is sticky, so a single NaN
            # anywhere -- including at the anchor -- makes min_U NaN for the
            # rest of the run, the termination test compares against NaN, and
            # the ladder quits early. Measured: it stopped at E = -145 instead
            # of -173. Mapping non-finite to +inf keeps the reduction clean.
            min_U=jnp.minimum(state.min_U,
                              jnp.min(jnp.where(jnp.isfinite(U), U, jnp.inf))),
            step=drift_update(state.step, acceptance - acc_target, step_gain),
            # "frozen" holds the matrix warmed at E_0; "unit" never has one.
            # Either way the step size is the only thing adapting down the
            # ladder. The case for not adapting: the score metric is
            # re-estimated from the SAME branched cloud it preconditions, so
            # estimate and estimand move together -- a direction the ensemble
            # has collapsed in gets a smaller sd, which lets it collapse
            # further. The case against: the geometry genuinely drifts down a
            # quenched path, and a scalar step can only absorb the isotropic
            # part of that drift.
            metric=(state.metric if metric_mode in ("frozen", "unit")
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
        # the volume ratio is an average under the CARRIED weights, not under a
        # uniform cloud: after a level without resampling the walkers are not
        # equally weighted, and treating them as such biases Lambda.
        # A NaN weight is a dead walker, not a poison pill. One NaN makes
        # logsumexp NaN, which makes the whole normalised vector NaN, which
        # marks every walker dead and replaces the population from NaN weights.
        # Map it to -inf first: outside the level set is exactly what it means.
        # a walker whose coordinates have gone NaN is dead in the same sense
        # as one outside the level set, and must be caught here or it survives
        # every subsequent test
        bad = jnp.isnan(log_w) | ~jnp.isfinite(state.U)
        log_W = jnp.where(bad, -jnp.inf, state.log_W + log_w)
        log_lambda = state.log_lambda + logsumexp(log_W) - logsumexp(state.log_W)

        # running integral for the termination criterion only; the estimate is
        # assembled on the host at the end, by exact quadrature.
        dE = state.E - E_new
        ok = dE > 0
        term = jnp.where(
            ok,
            jnp.log(jnp.where(ok, 0.5 * dE, 1.0))
            + jnp.logaddexp(state.log_lambda - state.E, log_lambda - E_new),
            -jnp.inf,
        )

        # RESAMPLE ON DEGENERACY, NOT EVERY LEVEL. A full systematic redraw
        # every rung is the SMC convention and it is what kills minority modes:
        # over a ladder of thousands of levels, neutral drift extinguishes any
        # subpopulation long before the bottom, whatever its weight. On LJ38
        # the fcc funnel is a small fraction of the ensemble at the energy where
        # the funnels separate, so every-level resampling guarantees it is lost
        # and the run reports a single funnel no matter how deep it goes.
        # Carrying the weights and redrawing only when the ESS actually
        # collapses makes the redraw ~100x rarer and preserves the minority.
        key, key_branch, key_resample = jax.random.split(state.key, 3)
        log_W = log_W - logsumexp(log_W)
        w = jnp.exp(log_W)

        # BRANCH THE DEAD EVERY LEVEL. A walker above the new level carries
        # weight exactly zero, and without a redraw it never leaves: rho_E is
        # -inf there, so the kernel cannot move it and its coordinates go NaN.
        # Refill only those slots -- copy a survivor drawn by weight, split the
        # donor's weight across donor and copies, which is measure-preserving.
        # One stratum per DEAD slot, over the survivors' cdf. Taking the dead
        # slots' entries out of an N-stratum draw instead hands each donor to
        # the dead slot's array POSITION rather than to weight: measured, the
        # copies per particle then correlate 0.42 with the n_dead*w they should
        # match, against 0.9999 here. side="right" steps past the zero-width
        # intervals the dead leave in the cdf, so a dead walker is never its own
        # donor.
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

        # THE TRIGGER READS THE LINEAGE-GROUPED ESS, NOT THE PLAIN ONE.
        # Splitting a donor's weight halves its contribution to sum(w^2), so
        # the plain Kish ESS INFLATES with every branch and a trigger reading
        # it never fires: the population is then never refreshed, walkers
        # accumulate against the level boundary where the score
        # -nu grad U / (E - U) diverges, and the run dies of NaN partway down.
        # Grouped by ancestor, a walker and its copies count as the one sample
        # they are, so the trigger fires when diversity has actually collapsed.
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
            # the RATIO the schedule actually holds. With weights carried, the
            # increment's own ESS is a different number and reports the
            # schedule as missing a target it was never set.
            ess=jnp.exp(log_ess(log_W) - log_ess(state.log_W)),
            step_size=jnp.exp(new.step.average),
        )
        # the mutated cloud is the sample from rho_E; the resampled one belongs
        # to the next level. `track` is applied on device to ALL walkers, not
        # to the retained slice: a summary is a few floats per walker where the
        # positions are D, so the whole ladder is affordable in the summary and
        # hopeless in the coordinates.
        return new, info, LevelSample(
            walkers=state.particles[:n_keep],
            log_w=log_W,
            stat=(jnp.zeros((n, 0), state.particles.dtype) if track is None
                  else track(state.particles)),
        )

    @partial(jax.jit, static_argnames=("n",))
    def warmup(state: LadderState, n: int):
        """Nesterov dual averaging on the step size, at E_0, before any level.

        Ported from quenched_sampling (methods.warmup), which every paper run
        used and this rewrite silently dropped -- the third such divergence
        after the resampling cadence and the log1mexp weights. The ladder's own
        controller is constant-gain and MULTIPLICATIVE, so the step it starts
        from must be within a factor of a few; dual averaging converges ON the
        target acceptance from an arbitrary start, and the averaged iterate is
        insensitive to any single noisy acceptance estimate. The failure this
        prevents is silent: a step orders off scale gives acceptance 0.000,
        frozen walkers, a min U that never improves, and an early exit whose
        evidence looks converged.

        The metric keeps warming through the same passes, exactly as before.
        The recursion itself lives in `qes.adapt`, because the tempered arm
        warms up the same way and "matched in everything but the path" has to
        be a fact about the code rather than a claim in a docstring.
        """

        def body(carry, key):
            st, da = carry
            # warmup runs at E_0 on the freshly resampled anchor cloud: every
            # walker carries the same weight and no branching has happened yet,
            # so the metric sees uniform weights at full lineage ESS.
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
    n_steps_first: int | None = None,
    n_first: int = 2,
    heavy_above: float | None = None,
    step_size: float = 0.1,
    acc_target: float = 0.574,
    step_gain: Gain = Gain(rate=0.5, kappa=0.0, floor=0.0, poly=0.15),
    metric_gain: Gain = Gain(rate=1.0, kappa=0.0, floor=0.0, poly=1.0),
    max_levels: int = 20_000,
    E_stop: float | None = None,
    metric_mode: str = "score",
    n_keep: int = 40,
    track: Callable | None = None,
    resample_ess: float = 0.5,
    n_leapfrog: int | None = None,
    verbose: int = 0,
) -> Result:
    """Estimate log Z by a quenched ladder of soft microcanonical levels.

    U_fn, log_prior, sample_prior
        U = -log L and log pi at one point (normalised, and smooth), and
        (key, n) -> (n, D) prior draws.
    nu
        Level softness; 2 throughout, and the useful range does not scale with D.
    n_walkers, n_steps
        Budgets, set by hand. n_steps need not grow with dimension.
    target_ess
        The ESS fraction each level's weights must retain: the only schedule.
    dlogz
        Termination depth; must exceed the latent heat of any transition.
    n_init
        Prior draws for the anchor, default 10 * n_walkers.
    n_start
        Place E_0 at the n_start-th smallest prior energy rather than at the
        ESS target. Use it when U has a divergent core, where max(U) is
        meaningless and the default search starts above every physical scale.
    n_steps_first, n_first, heavy_above
        Mutation steps for the first `n_first` rungs, or for every rung with
        E > heavy_above, then back to n_steps. The window form is the one that
        matters on a condensing system: equilibrate while the cluster is still
        liquid and its basins still interconvert, then quench from an ensemble
        that already carries the right split. Below the barrier the basins are
        disconnected and no mutation budget can move probability between them.
        The two funnels are disconnected below the barrier, so the population
        keeps whichever basins it holds when they separate; buying a
        well-equilibrated ensemble at the top is therefore worth far more than
        the same gradients spent uniformly down a ladder of thousands of
        levels. Costs one extra compilation.
    n_warmup
        Mutate-and-adapt passes at E_0. They set the step size, warm the score
        metric, and equilibrate walkers that resampling delivered only in
        distribution.
    step_gain, metric_gain
        The two adaptation gains (`qes.adapt`).
    metric_mode
        "score" tracks the diagonal metric down the ladder; "score_rows" tracks
        it the way the code did before the lineage-ESS correction, averaging
        the score pattern unweighted and shrinking by the number of array rows
        rather than by the number of lineages; "frozen" warms it
        at E_0 and then holds it; "unit" is the identity throughout and is
        never estimated at all. Under the latter two the step size is the only
        adapted quantity, and the ladder is a plain isotropic MALA with a
        controller. "unit" also changes what the warmup does: there is no
        matrix to warm, so the passes only set the step size.

    Only the metric and the scalar step size are adapted during the descent.
    """
    if metric_mode not in ("score", "score_rows", "frozen", "unit"):
        raise ValueError(f"metric_mode={metric_mode!r} is not one of "
                         "'score', 'score_rows', 'frozen', 'unit'")
    n_init = 10 * n_walkers if n_init is None else n_init
    if n_init < n_walkers:
        raise ValueError(f"n_init={n_init} cannot furnish n_walkers={n_walkers}")

    key, key_anchor, key_ladder = jax.random.split(key, 3)
    x0, E0, log_lambda0, log_top = anchor(
        key_anchor, U_fn, sample_prior, nu, n_walkers, n_init, n_start
    )
    level, warmup = _build_level(
        U_fn, log_prior, nu, n_steps, target_ess, acc_target, step_gain,
        metric_gain, n_keep, track, resample_ess, n_leapfrog, metric_mode,
    )
    level_first = level if n_steps_first is None else _build_level(
        U_fn, log_prior, nu, n_steps_first, target_ess, acc_target, step_gain,
        metric_gain, n_keep, track, resample_ess, n_leapfrog, metric_mode,
    )[0]

    # the cloud seeds the metric shape for the first pass only; the first
    # adaptation replaces it with the score estimate.
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
    ess, steps, accs, snapshots = [], [], [], []
    stats, log_ws = [], []
    E_prev = float(E0)
    for k in range(max_levels):
        heavy = (k < n_first if heavy_above is None
                 else float(state.E) > heavy_above)
        state, info, sample = (level_first if heavy else level)(state)
        info = jax.device_get(info)  # one host transfer per level

        # no progress: the spacing has fallen below the resolution of E itself,
        # so continuing costs gradients and an inverted interval would poison
        # the evidence with a NaN. A convergence criterion, not a guard.
        if not (info.E < E_prev):
            break

        Es.append(float(info.E))
        log_lambdas.append(float(info.log_lambda))
        ess.append(float(info.ess))
        steps.append(float(info.step_size))
        accs.append(float(info.acceptance))
        snapshots.append(np.asarray(sample.walkers))
        # always: the pooled posterior needs the within-rung weights whether or
        # not a summary is being tracked. One scalar per walker per rung.
        log_ws.append(np.asarray(sample.log_w))
        if track is not None:
            stats.append(np.asarray(sample.stat))
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
        if E_stop is not None and float(info.E) <= E_stop:
            break

    # host records in the sampler's own precision: `float()` returns a Python
    # double, and letting numpy infer from that would quietly do the quadrature
    # at a precision the ladder never ran at.
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

    # a ladder that never moved still returns a number, and the number looks
    # converged: frozen walkers never improve min U, the dissection drives E
    # onto that floor and termination fires within a few levels.
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
        stats=np.asarray(stats) if stats else np.empty((0, 0, 0)),
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
    """Equally weighted posterior draws from the whole path, not just its last
    cloud: rung k carries quadrature weight b_k = Lambda(E_k) e^-E_k dE_k from
    the augmentation, walker j within it carries a_kj, and the pooled weight is
    q_kj proportional to b_k a_kj. Drawing uniformly within a rung is wrong
    wherever branching has split a donor's weight across its copies.
    Discretisation in E is the only approximation."""
    snapshots = result.snapshots
    if snapshots.size == 0:
        raise ValueError("no snapshots were stored; run with n_keep > 0")
    k, n_keep = snapshots.shape[:2]
    Es, log_lambdas = result.Es[:k], result.log_lambdas[:k]
    dE = np.abs(np.gradient(Es)) if k > 1 else np.ones_like(Es)
    log_b = log_lambdas - Es + np.log(np.clip(dE, np.finfo(dE.dtype).tiny, None))

    log_a = result.log_weights[:k, :n_keep]
    a = np.exp(log_a - log_a.max(axis=1, keepdims=True))
    a /= a.sum(axis=1, keepdims=True)          # renormalised over the retained
    q = np.exp(log_b - log_b.max())[:, None] * a
    q = (q / q.sum()).ravel()

    draws = np.asarray(jax.random.choice(key, q.size, (n,), p=jnp.asarray(q)))
    return snapshots.reshape(-1, snapshots.shape[-1])[draws]


# ---------------------------------------------------------------------------
# quadrature
# ---------------------------------------------------------------------------
def log_integrate(E, log_f) -> float:
    """log int f dE over a strictly descending grid, f given by its logarithm.

    Exact where log f is linear in E, which is the relevant limit (e^-E is, and
    log Lambda nearly is within one level):

        int_b^a f dE = (f(a) - f(b)) / m,   m = (log f(a) - log f(b)) / (a - b),

    with the trapezoid taken where the interval is flat, at a threshold read
    from the dtype. Non-positive intervals are dropped: deep in a ladder the
    spacing falls below the resolution of E itself, and a zero-width interval
    contributes no area anyway.
    """
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
