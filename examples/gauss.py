"""Isotropic Gaussian: unimodal, no transition. The scaling and validation
target, exact at every D.

pi = N(0, I_D) and L(x) = exp(-|x|^2 / 2 sigma^2) with sigma = 0.5, so with
tau^2 = sigma^2/(1+sigma^2) = 0.2,

    log Z = (D/2) log tau^2,    posterior = N(0, tau^2 I),    min U = 0.

Paper configuration (tab:analytic): N = 1000, target ESS 0.95,
n_steps = 2^ceil(log2 max(16, sqrt(D))), dlogz = -3 fixed in D (no latent
heat -- the depth is a tolerance, not a property of D), and metric_mode="unit":
the archived TPU runs all carry mass=identity, since on an isotropic target the
score metric has nothing to say and the identity is the controlled choice.

    uv run python examples/gauss.py --D 100
    uv run python examples/gauss.py --D 10000 --method tempered
"""
import argparse
import math
import time
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

import qes


@dataclass(frozen=True)
class Gauss:
    D: int
    U_fn: Callable
    log_prior: Callable
    sample_prior: Callable
    log_Z: float
    post_sd: float
    dlogz: float


def gauss(D: int, sigma: float = 0.5) -> Gauss:
    tau2 = sigma**2 / (1.0 + sigma**2)

    def U_fn(x):
        return jnp.sum(x**2) / (2.0 * sigma**2)

    def log_prior(x):
        return -0.5 * jnp.sum(x**2) - (D / 2) * jnp.log(2 * jnp.pi)

    def sample_prior(key, n):
        return jax.random.normal(key, (n, D))

    return Gauss(D, U_fn, log_prior, sample_prior,
                 log_Z=(D / 2) * math.log(tau2),
                 post_sd=math.sqrt(tau2), dlogz=-3.0)


def paper_n_steps(D: int) -> int:
    return 2 ** math.ceil(math.log2(max(16.0, math.sqrt(D))))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--D", type=int, default=100)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--walkers", type=int, default=1000)
    p.add_argument("--steps", type=int, default=None,
                   help="default: the paper rule, 2^ceil(log2 max(16, sqrt D))")
    p.add_argument("--nu", type=float, default=2.0)
    p.add_argument("--target-ess", type=float, default=0.95)
    p.add_argument("--metric", default="unit",
                   choices=("unit", "score", "frozen"),
                   help="identity mass, score-adapted down the ladder, or "
                        "score-warmed then held fixed")
    p.add_argument("--method", choices=("qes", "tempered"), default="qes")
    p.add_argument("--posterior-draws", type=int, default=20_000)
    p.add_argument("--verbose", type=int, default=0)
    args = p.parse_args()

    t = gauss(args.D)
    n_steps = args.steps or paper_n_steps(args.D)
    print(f"gauss: D={t.D}  exact log Z {t.log_Z:.4f}  post sd {t.post_sd:.4f}  "
          f"steps {n_steps}  metric {args.metric}  dtype {jnp.zeros(1).dtype}\n")

    log_Zs = []
    for seed in range(args.seeds):
        key_run, key_post = jax.random.split(jax.random.key(seed))
        start = time.perf_counter()
        if args.method == "qes":
            r = qes.run(key_run, t.U_fn, t.log_prior, t.sample_prior,
                        nu=args.nu, n_walkers=args.walkers, n_steps=n_steps,
                        target_ess=args.target_ess, dlogz=t.dlogz,
                        metric_mode=args.metric, verbose=args.verbose)
            draws = qes.posterior_sample(key_post, r, args.posterior_draws)
            rungs = r.n_levels
        else:
            r = qes.tempered.run(key_run, t.U_fn, t.log_prior, t.sample_prior,
                                 n_walkers=args.walkers, n_steps=n_steps,
                                 target_ess=args.target_ess,
                                 metric_mode=args.metric, verbose=args.verbose)
            draws = qes.tempered.posterior_sample(key_post, r,
                                                  args.posterior_draws)
            rungs = r.n_stages
        wall = time.perf_counter() - start
        log_Zs.append(r.log_Z)
        # every coordinate is exchangeable: pool them for the moment check
        sd_hat = float(np.std(draws))
        print(f"seed {seed}:  dlogZ {r.log_Z - t.log_Z:+.4f}   rungs {rungs}   "
              f"grads {r.grad_evals:.3g}   acc {r.acceptance:.3f}   "
              f"post sd {sd_hat:.4f}/{t.post_sd:.4f}   {wall:.1f}s", flush=True)

    if args.seeds > 1:
        lz = np.asarray(log_Zs)
        lm = lz.max() + np.log(np.exp(lz - lz.max()).mean())
        print(f"\nover {args.seeds} seeds:  bias {lm - t.log_Z:+.4f}  "
              f"spread {lz.std():.4f}")


if __name__ == "__main__":
    main()
