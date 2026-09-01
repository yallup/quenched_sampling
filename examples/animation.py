"""The two paths, animated on the asymmetric double well:

    U(x) = h((x/a)^2 - 1)^2 + c x - min,   a = 1.5, h = 4, c = 0.4,

two deep phases with unequal floors, under a N(0, 2^2) prior.  Column kinds,
composable via animate():

  qes  the soft nu = 2 level ensemble, quenching the level E to depth --
       basin filled to E, rho_E, and the score nu |grad U| / (E - U),
       informative throughout the level set.
  smc  tempered SMC annealing beta from 0 to 1 -- the whole path of
       rho_beta over (x, beta), the current slice, and the score
       beta |grad U|, which never sees the second phase's floor.
  ns   nested sampling's hard constraint (nu = 0), whose score is zero
       inside the level set and a wall at its edge.

Both columns open AT THE PRIOR: beta = 0 is the prior exactly, and the level
ladder opens at E = 100, where rho_E is the prior to within a few percent
(the tilt is e^{-nu U / E}).  A short geometric intro carries E down to the
working range -- equal decrements of log G_nu in the E >> U regime, where
log G_nu ~ nu log E, just at a faster clip -- and from there levels pace by
equal decrements of log G_nu and beta by equal decrements of log Z_beta over
the same frame count, so all columns run in lockstep.

    uv run python examples/animation.py    # examples/figures/qes_smc.gif
"""
import subprocess
import tempfile
from pathlib import Path

FIGURES = Path(__file__).resolve().parent / "figures"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm
from tueplots import bundles

plt.rcParams.update(bundles.tmlr2023())
plt.rcParams.update({"text.usetex": True,
                     "text.latex.preamble": r"\usepackage{amsmath,amssymb}"})
COL = {"ns": "#D55E00", "qes": "#0072B2", "smc": "#009E73"}
TITLE = {"ns": r"nested sampling: $\pi\,\mathbf{1}\{U<E\}$ \quad ($\nu = 0$)",
         "qes": r"QES: $\pi\,(E-U)_+^{\nu}$ \quad ($\nu = 2$)",
         "smc": r"tempered SMC: $\pi\,e^{-\beta U}$ \quad ($\beta: 0 \to 1$)"}

NU = 2.0
A, H, C = 1.5, 4.0, 0.4
x = np.linspace(-3.4, 3.4, 3001)
_raw = H * ((x / A) ** 2 - 1) ** 2 + C * x
U = _raw - _raw.min()
dU = np.gradient(U, x)
prior = norm.pdf(x, scale=2)
QUANT = np.linspace(0.05, 0.95, 16)
NF, YM = 270, 9.0

E_HI, E_LO, E_LO_NS, E_PRIOR, N_INTRO = 10.0, 0.32, 1.0, 100.0, 40
Eg = np.linspace(E_LO, E_HI, 1200)
_G = np.array([np.trapezoid(prior * np.clip(E - U, 0, None) ** NU, x)
               for E in Eg])
_logG = np.log(_G)
# ONE ladder, shared by both level columns: identical level sets at every
# frame.  A geometric intro from E_PRIOR opens at the prior, then the working
# ladder runs from E_HI down to the common floor E_LO_NS.
_loNS = float(np.interp(E_LO_NS, Eg, _logG))
_work = np.interp(np.linspace(_loNS, _logG[-1], NF), _logG, Eg)[::-1]
_intro = np.geomspace(E_PRIOR, _work[0], N_INTRO, endpoint=False)
E_SCHED = np.concatenate([_intro, _work])
NF_TOT = len(E_SCHED)
# beta spans the same frames, by equal decrements of log Z_beta, so frame 0
# is beta = 0: the prior itself.
Bg = np.linspace(0.0, 1.0, 1200)
_Zb = np.array([np.trapezoid(prior * np.exp(-b * U), x) for b in Bg])
_logZ = np.log(_Zb)
B_SCHED = np.interp(-np.linspace(_logZ[0], _logZ[-1], NF_TOT), -_logZ, Bg)

Bs = np.linspace(0, 1, 240)
P_BETA = np.array([prior * np.exp(-b * U) for b in Bs])
P_BETA = P_BETA / P_BETA.max(axis=1, keepdims=True)
CMAP_SMC = LinearSegmentedColormap.from_list("smc", ["white", COL["smc"]])


def dens_level(E, nu):
    d = np.where(U < E, prior * (1.0 if nu == 0 else
                                 np.clip(E - U, 0, None) ** nu), 0.0)
    Z = np.trapezoid(d, x)
    return d / Z if Z > 0 else d


def dens_beta(b):
    d = prior * np.exp(-b * U)
    return d / np.trapezoid(d, x)


def edges(E):
    m = (U < E).astype(int)
    j = np.flatnonzero(np.diff(m))
    return 0.5 * (x[j] + x[j + 1])


def animate(cols, outname):
    n = len(cols)
    fig, ax = plt.subplots(3, n, figsize=({1: 3.6, 2: 6.2, 3: 8.4}[n], 3.68),
                           sharex=True, squeeze=False)
    texts = {}
    for i, kind in enumerate(cols):
        a0, a1, a2 = ax[0, i], ax[1, i], ax[2, i]
        right = kind == "smc" and n > 1
        if kind == "smc":
            a0.pcolormesh(x, Bs, P_BETA, cmap=CMAP_SMC, rasterized=True,
                          shading="auto")
            a0.set_ylim(0, 1)
            a0.set_yticks([0, 0.5, 1])
            if right:
                a0.yaxis.tick_right()
                a0.yaxis.set_label_position("right")
            a0.set_ylabel(r"$\beta$")
        else:
            a0.plot(x, U, color="0.3", lw=0.9)
            a0.set_ylim(-0.4, 7.3)
            a0.set_yticks([])
        a1.plot(x, prior / prior.max(), color="0.5", lw=0.7, ls=":")
        a1.set_ylim(0, 1.12)
        a1.set_yticks([])
        a2.plot(x, np.abs(dU), color="0.5", lw=0.7, ls=":")
        a2.set_ylim(-0.5, YM)
        a2.set_yticks([])
        a2.axhline(0, color="0.85", lw=0.5, zorder=0)
        a2.set_xlabel(r"$x$")
        for a_ in (a0, a1, a2):
            a_.set_xlim(-3.4, 3.4)
            a_.set_xticks([])
        a0.set_title(TITLE[kind])
        if kind == "smc":
            for a_, lab in ((a1, r"$\rho_\beta(x)$"),
                            (a2, r"$\beta\,|\nabla U|$")):
                if right:
                    a_.yaxis.set_label_position("right")
                a_.set_ylabel(lab)
            texts["beta"] = a1.text(-3.25, 0.98, "", fontsize=7, color="0.2")
        elif i == 0:
            a0.set_ylabel(r"$U(x)$")
            a1.set_ylabel(r"$\rho_E(x)$")
            a2.set_ylabel(r"$|\nabla\log(E-U)_+^{\nu}|$")
        if kind == "ns" or (kind == "qes" and "ns" not in cols):
            texts["E"] = a0.text(-3.25, 0.35, "", fontsize=7, color="0.2")

    dyn = []

    def draw(frame):
        nonlocal dyn
        for art in dyn:
            art.remove()
        dyn = []
        E_shared = float(E_SCHED[frame])
        b = float(B_SCHED[frame])
        for i, kind in enumerate(cols):
            a0, a1, a2 = ax[0, i], ax[1, i], ax[2, i]
            c = COL[kind]
            if kind == "smc":
                dyn.append(a0.axhline(b, color="0.2", lw=0.8, ls="--"))
                d = dens_beta(b)
                dp = d / d.max()
                dyn.append(a1.fill_between(x, dp, color=c, alpha=0.15, lw=0))
                dyn.extend(a1.plot(x, dp, color=c, lw=0.9))
                sm = np.clip(b * np.abs(dU), 0, YM + 1)
                dyn.append(a2.fill_between(x, 0, sm, color=c, alpha=0.15,
                                           lw=0))
                dyn.extend(a2.plot(x, sm, color=c, lw=0.9))
                texts["beta"].set_text(rf"$\beta = {b:.2f}$")
            else:
                nu = 0.0 if kind == "ns" else NU
                E = E_shared
                inside = U < E
                ed = edges(E)
                dyn.append(a0.fill_between(x, U, E, where=inside, color=c,
                                           alpha=0.15, lw=0))
                d = dens_level(E, nu)
                dp = d / d.max()
                dyn.append(a1.fill_between(x, dp, color=c, alpha=0.15, lw=0))
                dyn.extend(a1.plot(x, dp, color=c, lw=0.9))
                if kind == "ns":
                    for e in ed:
                        h = max(float(np.interp(e - 3e-3, x, dp)),
                                float(np.interp(e + 3e-3, x, dp)))
                        dyn.append(a1.vlines(e, 0, h, color=c, lw=1.1))
                    for k in range(len(ed) // 2):
                        seg = inside & (x >= ed[2 * k]) & (x <= ed[2 * k + 1])
                        dyn.extend(a2.plot(x[seg],
                                           np.zeros(int(seg.sum())),
                                           color=c, lw=0.9))
                    for e in ed:
                        dyn.extend(a2.plot([e, e], [0, 0.93 * YM], color=c,
                                           lw=0.9))
                        dyn.extend(a2.plot([e], [0.93 * YM], marker="^",
                                           color=c, ms=3.5))
                else:
                    mag = np.clip(nu * np.abs(dU) /
                                  np.where(inside, E - U, np.nan), 0, YM + 1)
                    dyn.append(a2.fill_between(x, 0, np.nan_to_num(mag),
                                               where=inside, color=c,
                                               alpha=0.15, lw=0))
                    dyn.extend(a2.plot(x, mag, color=c, lw=0.9))
                if "E" in texts:
                    texts["E"].set_text(rf"$E = {E:.2f}$")
            cdf = np.cumsum(d); cdf /= cdf[-1]
            q = np.interp(QUANT, cdf, x)
            dyn.extend(a1.plot(q, np.full(len(q), 0.034), ".", color=c,
                               ms=2.4))
        return dyn

    anim = animation.FuncAnimation(fig, draw, frames=NF_TOT, blit=False)
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / outname
    # render at full quality, then palette-quantise to a compact GIF; the
    # intermediate never leaves the temp dir.
    scale = "fps=10,scale=640:-1:flags=lanczos"
    with tempfile.TemporaryDirectory() as td:
        raw, pal = Path(td) / "raw.mp4", Path(td) / "pal.png"
        anim.save(raw, writer=animation.FFMpegWriter(
                      fps=18, extra_args=["-crf", "23", "-preset", "slow"]),
                  dpi=200)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw,
                        "-vf", f"{scale},palettegen", pal], check=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-i", pal,
                        "-filter_complex",
                        f"{scale}[s];[s][1:v]paletteuse=dither=bayer:"
                        "bayer_scale=4", out], check=True)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    animate(("qes", "smc"), "qes_smc.gif")
