import jax
import jax.numpy as jnp
import numpy as np
import pytest

import qes
from qes.adapt import Gain, drift_init, drift_update
from qes.qes import anchor
from spike_slab import spike_slab


def test_weights_outside_both_levels_are_zero_not_nan():
    U = jnp.array([0.0, 5.0])
    logw = qes.level_logw(U, jnp.asarray(1.0), jnp.asarray(0.5), 2.0)
    assert jnp.isneginf(logw[1])
    assert jnp.isfinite(logw[0])


def test_next_level_hits_the_target_ess():
    U = 2.0 * jax.random.normal(jax.random.key(0), (500,))
    E = jnp.max(U) + 1.0
    E_new = qes.next_level(U, E, 2.0, 0.8)
    ess = qes.ess_fraction(qes.level_logw(U, E, E_new, 2.0))
    assert E_new < E
    assert 0.8 <= float(ess) < 0.85


def test_next_level_survives_a_level_far_above_min_u():
    """The bracket end must stay clear of min U after the round trip E - delta,
    even when |E| >> |min U|; otherwise every weight is -inf and the solver
    stalls at delta = 0."""
    U = jnp.array([1.3947069685008335, 3.0, 7.0, 10.6])
    E = jnp.asarray(1.7768600459545e1)
    E_new = qes.next_level(U, E, 2.0, 0.8)
    assert float(E_new) < float(E)
    assert jnp.isfinite(qes.level_logw(U, E, E_new, 2.0)).any()


def test_log_integrate_is_exact_on_an_exponential():
    E = np.linspace(10.0, 0.0, 11)
    exact = np.log(1 - np.exp(-10.0))
    assert qes.log_integrate(E, -E) == pytest.approx(exact, abs=1e-6)


def test_log_integrate_takes_the_flat_limit():
    E = np.linspace(4.0, 0.0, 5)
    flat = np.zeros_like(E)
    assert qes.log_integrate(E, flat) == pytest.approx(np.log(4.0), abs=1e-6)


def test_drift_at_unit_gain_is_replacement():
    """gain 1 with poly 1 is "measure at the previous level, freeze it here"."""
    state = drift_init(jnp.zeros(3))
    signal = jnp.array([2.0, 2.0, 2.0]) - state.value
    state = drift_update(state, signal, Gain(rate=1.0, kappa=0.0, floor=0.0, poly=1.0))
    assert np.allclose(state.value, 2.0)
    assert np.allclose(state.average, 2.0)


def test_anchor_places_the_first_level_at_ess_equal_to_n_walkers():
    target = spike_slab(10)
    n_walkers, n_init = 100, 1000
    x, E0, log_lambda0, _ = anchor(
        jax.random.key(0), target.U_fn, target.sample_prior, 2.0, n_walkers, n_init
    )
    draws = target.sample_prior(jax.random.key(1), n_init)
    U = jax.vmap(target.U_fn)(draws)
    logw = qes.log_phi(E0 - U, 2.0)
    ess = float(jnp.exp(2 * jax.scipy.special.logsumexp(logw)
                        - jax.scipy.special.logsumexp(2 * logw)))
    assert x.shape == (n_walkers, target.D)
    assert n_walkers / 2 < ess < 2 * n_walkers
    assert jnp.isfinite(log_lambda0)


def test_spike_slab_evidence_and_phases():
    """End to end against the closed form, across the transition."""
    target = spike_slab(10)
    key_run, key_post = jax.random.split(jax.random.key(0))
    result = qes.run(
        key_run,
        target.U_fn,
        target.log_prior,
        target.sample_prior,
        n_walkers=200,
        n_steps=16,
        dlogz=target.dlogz,
    )
    draws = qes.posterior_sample(key_post, result, 5000)

    assert not result.degenerate
    assert abs(result.log_Z - target.log_Z) < 1.0
    assert abs(result.min_U - target.min_U) < 0.5  # the spike is reached
    assert 0.7 < target.spike_fraction(draws) < 1.0
    assert 0.4 < result.acceptance < 0.7
    assert result.log_tail_bound - result.log_Z < target.dlogz + 1.0
    assert np.all(np.diff(result.Es) < 0)  # strictly descending


def test_tempered_misses_the_spike_by_exactly_its_evidence():
    """The structural failure of the tempered path: the deficit is not a tuning
    gap a finer ladder would close, it is the evidence of the mode never
    visited. Refining to ESS 0.999 converges the bias onto -log 10."""
    target = spike_slab(10)
    key_run, key_post = jax.random.split(jax.random.key(0))
    result = qes.tempered.run(
        key_run,
        target.U_fn,
        target.log_prior,
        target.sample_prior,
        n_walkers=200,
        n_steps=16,
        target_ess=0.99,
    )
    draws = qes.tempered.posterior_sample(key_post, result, 5000)

    assert not result.degenerate
    assert result.betas[-1] == 1.0
    assert target.spike_fraction(draws) < 0.05  # never visits the spike
    assert -3.0 < result.log_Z - target.log_Z < -1.5  # short by about log 10


def test_tempered_takes_the_prescribed_ladder_when_given_one():
    target = spike_slab(5)
    result = qes.tempered.run(
        jax.random.key(0),
        target.U_fn,
        target.log_prior,
        target.sample_prior,
        n_walkers=100,
        n_steps=8,
        n_beta=12,
    )
    assert result.n_stages == 12
    assert result.betas[-1] == 1.0
    assert np.allclose(np.diff(result.betas), 1.0 / 12, atol=1e-5)


def test_posterior_snapshot_keeps_weights_for_its_recorded_level(monkeypatch):
    """A transition must not attach next-level weights to the old-level cloud."""
    import importlib
    from types import SimpleNamespace

    ladder = importlib.import_module("qes.qes")

    # Identity is invariant and isolates accounting from MCMC randomness.
    def identity_mutation(*args):
        def mutate(key, x, E, step, sd, weights, ess):
            return x, jnp.asarray(0.574), jnp.zeros(x.shape[1])
        return mutate

    monkeypatch.setattr(ladder, "build_mutation", identity_mutation)
    level, _ = ladder._build_level(
        lambda x: x[0], lambda x: -jnp.sum(x * x) / 2,
        2.0, 0, 0.8, 0.574, Gain(), Gain(), 4, 0.5,
    )
    weights = jnp.array([0.1, 0.2, 0.3, 0.4])
    x = jnp.arange(4.0)[:, None]
    state = ladder.LadderState(
        key=jax.random.key(0), particles=x, log_W=jnp.log(weights),
        ancestor=jnp.arange(4), U=x[:, 0], E=jnp.asarray(5.0),
        log_lambda=jnp.asarray(0.0), log_integral=jnp.asarray(-jnp.inf),
        min_U=jnp.asarray(0.0), step=drift_init(jnp.asarray(-2.0)),
        metric=drift_init(jnp.zeros(1)),
    )
    new, _, sample = level(state)
    assert float(new.E) < 3.0  # last walker is excluded from the NEXT level
    np.testing.assert_allclose(sample.walkers, x)
    np.testing.assert_allclose(jax.nn.softmax(sample.log_w), weights, rtol=1e-6)

    # Match run()'s storage: one snapshot and both ladder endpoints.
    result = SimpleNamespace(
        snapshots=np.asarray(sample.walkers)[None],
        log_weights=np.asarray(sample.log_w)[None],
        Es=np.array([float(state.E), float(new.E)]),
        log_lambdas=np.array([float(state.log_lambda), float(new.log_lambda)]),
    )
    draws = ladder.posterior_sample(jax.random.key(1), result, 20000)
    frequencies = np.bincount(draws[:, 0].astype(int), minlength=4) / len(draws)
    np.testing.assert_allclose(frequencies, weights, atol=0.015, rtol=0)
