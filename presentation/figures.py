"""Figures de la présentation et de la note (presentation/figures/*.pdf et *.png), au format diapositive.

Lit les résultats de main.py et de main_reduction.py (resultats/*.npz) ; seule la figure de validation relance
quelques résolutions.

    python3 presentation/figures.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))                                    # rend les paquets importables
from chaleur.cas import CAS, M
from chaleur.solveurs import transitoire
from chaleur.monte_carlo import tirer, monte_carlo, statistiques
from reduction.lineaire import POD, SousEspaceActif, GalerkinLineaire

RESULTATS = RACINE / "resultats"
SORTIE = Path(__file__).resolve().parent / "figures"
NOMS = {"conductivite": "Conductivité aléatoire", "source": "Source aléatoire"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 7.5, "figure.dpi": 100})


def charger(nom):
    """Résultats Monte Carlo (main.py) et réduits (main_reduction.py) d'un cas."""
    return np.load(RESULTATS / f"monte_carlo_{nom}.npz"), np.load(RESULTATS / f"reduction_lineaire_{nom}.npz")


def sauver(fig, nom, legende=None, ncol=3):
    """Enregistre en pdf (Beamer) et png (note) ; legende : axe dont la légende est placée sous la figure."""
    if legende is not None:
        fig.legend(*legende.get_legend_handles_labels(), loc="lower center", ncol=ncol, frameon=False)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
    else:
        fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(SORTIE / f"{nom}.{ext}", dpi=200)
    plt.close(fig)


def zones(ax):
    for xi in (1 / 3, 2 / 3):
        ax.axvline(xi, color="0.6", ls=":", lw=0.8)


def fig_probleme():
    """Profils u(x, t) d'un matériau par cas : lissage rapide, coudes fixes aux interfaces (conductivité)."""
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.5), sharey=True)
    for a, nom in zip(ax, NOMS):
        mc, _ = charger(nom)
        x, t, nu, U = mc["x"], mc["t"], mc["nu"], mc["u"]
        i = np.argmax(np.ptp(np.log(nu), axis=1))                 # matériau le plus contrasté
        for j, tj in enumerate(t):
            if tj in (0, 0.02, 0.06, 0.1, 0.2, 1.0):
                a.plot(x, U[i, j], color=plt.cm.plasma(0.85 * tj ** 0.4), label=f"$t = {tj:g}$")
        zones(a)
        a.set(xlabel="$x$", title=f"{NOMS[nom].split()[0]}, $\\nu = ({', '.join(f'{v:.2f}' for v in nu[i])})$")
    ax[0].set(ylabel="$u(x, t)$", ylim=(-0.03, 1.32))
    ax[1].legend(ncol=3, loc="upper center")
    sauver(fig, "probleme")


def fig_spectres():
    """Spectres POD : diffusion (nos deux cas) contre transport d'une bosse (barrière de Kolmogorov)."""
    mc, _ = charger("source")
    x, t = mc["x"], mc["t"]
    centres = 0.2 + 0.6 * t                                        # bosse transportée à vitesse 0.6
    transport = np.exp(-((x[None, :] - centres[:, None]) / 0.08) ** 2)
    spectres = {NOMS[n]: charger(n)[1]["sv"] for n in NOMS}
    spectres["Transport d'une bosse (comparaison)"] = np.linalg.svd(transport - transport.mean(0), compute_uv=False)
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.5))
    for (nom, s), style in zip(spectres.items(), ("o-", "s-", "^--")):
        k = np.arange(1, 21)
        ax[0].semilogy(k, s[:20] / s[0], style, ms=3, label=nom)
        ax[1].semilogy(k[:-1], np.sqrt(np.maximum(1 - np.cumsum(s**2)[:19] / np.sum(s**2), 0)), style, ms=3, label=nom)
    ax[0].set(xlabel="mode $i$", ylabel="$\\sigma_i / \\sigma_1$", title="Valeurs singulières des snapshots", ylim=(1e-8, 2))
    ax[1].set(xlabel="dimension $d_u$", ylabel="erreur relative", title="Meilleure erreur linéaire (POD)", ylim=(1e-8, 2))
    sauver(fig, "spectres", legende=ax[0])


def fig_spectre_erreur():
    """Meilleure erreur linéaire (POD) selon d_u : diffusion (nos deux cas) contre transport d'une bosse (un panneau)."""
    mc, _ = charger("source")
    x, t = mc["x"], mc["t"]
    transport = np.exp(-((x[None, :] - (0.2 + 0.6 * t)[:, None]) / 0.08) ** 2)
    spectres = {NOMS[n]: charger(n)[1]["sv"] for n in NOMS}
    spectres["Transport d'une bosse"] = np.linalg.svd(transport - transport.mean(0), compute_uv=False)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    d = np.arange(1, 16)
    for (nom, s), style in zip(spectres.items(), ("o-", "s-", "^--")):
        err = np.sqrt(np.maximum(1 - np.cumsum(s**2) / np.sum(s**2), 0))[:len(d)]
        ax.semilogy(d[err > 0], err[err > 0], style, ms=3, label=nom)
    ax.set(xlabel="dimension $d_u$", ylabel="erreur relative", title="Meilleure erreur linéaire (POD)", ylim=(1e-10, 3))
    ax.legend(loc="lower left")
    sauver(fig, "spectre_erreur")


def fig_materiau():
    """Valeurs propres : ACP de phi(nu) (isotrope) contre sous-espace actif de l'équilibre."""
    fig, ax = plt.subplots(2, 1, figsize=(3.3, 3.4), sharex=True)
    for a, nom in zip(ax, NOMS):
        _, r = charger(nom)
        m = np.arange(1, M + 1)
        a.bar(m - 0.2, r["vp_acp"] / r["vp_acp"][0], 0.4, label="ACP de $\\varphi(\\nu)$")
        a.bar(m + 0.2, r["vp_actif"] / r["vp_actif"][0], 0.4, label="sous-espace actif")
        a.set(yscale="log", xticks=m, ylabel="v. p. / max", title=NOMS[nom])
    ax[1].set_xlabel("direction")
    sauver(fig, "materiau", legende=ax[0], ncol=2)


def fig_erreurs():
    """Erreur relative moyenne sur les matériaux non vus selon d_u : projection, réduit d_nu = 2 et d_nu = 3."""
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.5))
    for a, nom in zip(ax, NOMS):
        _, r = charger(nom)
        du, tab = r["dims_u"], r["tableau"]                        # (d_nu, d_u, proj/réduit/équilibre, moy/max)
        a.semilogy(du, tab[0, :, 0, 0], "o-", color="k", label="projection (meilleure possible)")
        for k, dn in enumerate(r["dims_nu"]):
            a.semilogy(du, tab[k, :, 1, 0], "s--" if k == 0 else "^-", label=f"modèle réduit, $d_\\nu = {dn}$")
        a.set(xlabel="$d_u$", title=NOMS[nom], xticks=du)
    ax[0].set_ylabel("erreur relative moyenne")
    sauver(fig, "erreurs", legende=ax[0])


def fig_plan_latent():
    """Plan latent (z_1, z_2) : trajectoires réduites depuis z0, équilibres z*(eta), modèle complet encodé."""
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.7))
    for a, nom in zip(ax, NOMS):
        mc, r = charger(nom)
        test = slice(int(r["n_app"]), None)
        eta, Zf, Zeq = r["eta"][test], r["z_fin"], r["z_eq"][test]
        Zvrai = (mc["u"][test] - r["ubar"]) @ r["V"]
        norme = plt.Normalize(eta[:, 0].min(), eta[:, 0].max())
        for i in range(len(Zf)):
            a.plot(Zf[i, :, 0], Zf[i, :, 1], color=plt.cm.viridis(norme(eta[i, 0])), lw=0.7)
        sc = a.scatter(Zeq[:, 0], Zeq[:, 1], c=eta[:, 0], norm=norme, s=5, cmap="viridis", zorder=3)
        for i in range(3):
            a.plot(Zvrai[i, :, 0], Zvrai[i, :, 1], "k--", lw=0.9, label="$E_u(u(t))$, modèle complet" if i == 0 else None)
        a.plot(*r["z0"], "r*", ms=10, label="$z_0 = E_u(u_0)$")
        fig.colorbar(sc, ax=a, label="$\\eta_1$")
        a.set(xlabel="$z_1$", ylabel="$z_2$", title=NOMS[nom])
    sauver(fig, "plan_latent", legende=ax[0], ncol=2)


def fig_statistiques():
    """E[T(t)] ± écart-type et loi de T* : Monte Carlo (500 résolutions complètes) contre modèle réduit."""
    fig, ax = plt.subplots(2, 2, figsize=(6.2, 3.3))
    for ligne, nom in zip(ax, NOMS):
        mc, r = charger(nom)
        t = mc["t"]
        T_mc = mc["u"].mean(axis=2)
        T_rom = (r["ubar"] + r["z"] @ r["V"].T).mean(axis=2)
        T_eq_mc, T_eq_rom = r["T_eq_mc"], (r["ubar"] + r["z_eq"] @ r["V"].T).mean(axis=1)
        for T, lab, style in ((T_mc, "Monte Carlo", "-"), (T_rom, "réduit", "--")):
            moy, ect, _ = statistiques(T)
            l, = ligne[0].plot(t, moy, style, label=f"$E[T]$, {lab}")
            ligne[0].fill_between(t, moy - ect, moy + ect, color=l.get_color(), alpha=0.15)
        ligne[0].set(ylabel="$T(t)$", title=f"{NOMS[nom]} : $E[T(t)]$ ± écart-type")
        bins = np.histogram_bin_edges(np.concatenate([T_eq_mc, T_eq_rom]), 25)
        ligne[1].hist(T_eq_mc, bins, density=True, alpha=0.5, label="Monte Carlo")
        ligne[1].hist(T_eq_rom, bins, density=True, histtype="step", lw=1.5, label="réduit")
        ligne[1].set(title=f"{NOMS[nom]} : loi de $T^*$")
    ax[1, 0].set_xlabel("$t$")
    ax[1, 1].set_xlabel("$T^*$")
    ax[0, 0].legend()
    ax[0, 1].legend()
    sauver(fig, "statistiques")


def fig_validation():
    """Avec d_u = N et d_nu = M, le modèle réduit redonne le modèle complet (conductivité, matériau non vu)."""
    cas = CAS["conductivite"]
    pb, t = cas.probleme(), np.linspace(0, 1, 51)
    nus = tirer([cas.loi] * M, 20, graine=10)
    rom = GalerkinLineaire(pb, POD(monte_carlo(lambda nu: transitoire(pb, nu, t), nus), len(pb.x)),
                           SousEspaceActif(pb, nus, M, cas.phi, cas.phi_inv))
    nu = tirer([cas.loi] * M, 1, graine=99)[0]
    u_ref = transitoire(pb, nu, t, rtol=1e-10, atol=1e-12)
    u_rom = rom.pod.decoder(rom.trajectoire(rom.materiau.encoder(nu), t))
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    for i in (0, 1, 5, 50):
        l, = ax.plot(pb.x, u_ref[i], lw=1, label=f"$t = {t[i]:g}$")
        ax.plot(pb.x[::3], u_rom[i, ::3], "o", ms=4, mfc="none", color=l.get_color())
    zones(ax)
    ax.set(xlabel="$x$", ylabel="$u$", ylim=(-0.03, 1.32), title="complet (traits), réduit $d_u = N$ (cercles)")
    ax.legend(ncol=2, loc="upper center")
    sauver(fig, "validation")


if __name__ == "__main__":
    SORTIE.mkdir(exist_ok=True)
    for f in (fig_probleme, fig_spectres, fig_spectre_erreur, fig_materiau, fig_erreurs, fig_plan_latent, fig_statistiques, fig_validation):
        f()
        print(f"{f.__name__} : OK")
