"""Validation du Monte Carlo sur des cas dont l'espérance est connue : conductivité OU source aléatoire"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))                                    # rend le paquet chaleur importable
from chaleur.modele import Chaleur1D, zones
from chaleur.solveurs import transitoire, stationnaire
from chaleur.monte_carlo import tirer, monte_carlo, statistiques


def compare(nom, moy, ic, exacte):
    """Affiche l'écart à la valeur exacte et sa compatibilité avec l'IC à 95 %."""
    ecart = abs(moy - exacte)
    print(f"{nom} : MC = {moy:.5f} ± {ic:.5f}, exacte = {exacte:.5f}, écart = {ecart:.1e}, "
          f"compatible IC 95 % : {'oui' if ecart <= ic else 'non'}")
    assert ecart <= 3 * ic / 1.96, "écart supérieur à 3 sigma / sqrt(n)"   # 99.7 % : évite les faux échecs


def cas_stationnaire(ax):
    """Conductivité aléatoire : k = nu ~ LogN(0, sigma^2) constante, f = 1, u(0) = u(1) = 0 : u = x(1-x) / (2 nu).

    Le schéma est exact (u polynomiale de degré 2) : seule reste l'erreur Monte Carlo. En x = 1/2, 1/nu ~ LogN(0, sigma^2) :
    E[u] = e^{sigma^2/2} / 8   et   Var[u] = (e^{2 sigma^2} - e^{sigma^2}) / 64.
    """
    sigma = 0.3
    lois = [stats.lognorm(s=sigma)]
    pb = Chaleur1D(19, lambda x, nu: np.full_like(x, nu[0]), lambda x, nu: np.ones_like(x))

    def u_milieu(nu):
        return stationnaire(pb, nu)[9]                             # noeud 9 : x = 1/2

    moy_ex = np.exp(sigma**2 / 2) / 8
    ect_ex = np.sqrt(np.exp(2 * sigma**2) - np.exp(sigma**2)) / 8

    # 1. Une estimation avec n = 1000 tirages
    moy, ect, ic = statistiques(monte_carlo(u_milieu, tirer(lois, 1000, graine=2026)))
    compare("Stationnaire, E[u(1/2)]", moy, ic, moy_ex)
    print(f"   écart-type : MC = {ect:.5f}, exact = {ect_ex:.5f}")
    assert abs(ect - ect_ex) < 0.1 * ect_ex, "écart-type trop éloigné de la valeur exacte"

    # 2. EQM de la moyenne empirique sur 20 graines : EQM = Var / n, donc pente -1 en log-log
    Ns = np.unique(np.logspace(1, 3, 10).astype(int))
    eqm = np.zeros(len(Ns))
    for graine in range(20):
        Y = monte_carlo(u_milieu, tirer(lois, Ns[-1], graine))
        eqm += (np.cumsum(Y)[Ns - 1] / Ns - moy_ex)**2 / 20      # moyennes cumulées
    pente = np.polyfit(np.log(Ns), np.log(eqm), 1)[0]
    print(f"   pente de l'EQM en log-log = {pente:.3f} (attendue : -1)")
    assert abs(pente + 1) < 0.25, "l'EQM ne décroît pas en 1/n"

    ax.loglog(Ns, eqm, "o", label="EQM sur 20 graines")
    ax.loglog(Ns, ect_ex**2 / Ns, "k--", label="$\\sigma^2 / n$")
    ax.set(xlabel="$n$", ylabel="EQM", title=f"Stationnaire : EQM de $u(1/2)$ (pente {pente:.2f})")
    ax.legend()


def cas_transitoire(ax):
    """k = nu ~ U(a, b) constant, f = 0, u0 = sin(pi x) : u = exp(-nu pi^2 t) sin(pi x),

    donc E[u(1/2, t)] = (exp(-a pi^2 t) - exp(-b pi^2 t)) / (pi^2 t (b - a)).
    """
    a, b, n = 0.5, 1.5, 300
    t = np.linspace(0, 0.3, 31)
    pb = Chaleur1D(49, lambda x, nu: np.full_like(x, nu[0]), lambda x, nu: 0 * x,
                   u0=lambda x: np.sin(np.pi * x))                 # h = 1/50 : noeud 24 en x = 1/2
    Y = monte_carlo(lambda nu: transitoire(pb, nu, t)[:, 24], tirer([stats.uniform(a, b - a)], n, graine=1))
    moy, ect, ic = statistiques(Y)
    c = np.pi**2 * t[1:]
    exacte = np.concatenate([[1.0], (np.exp(-a * c) - np.exp(-b * c)) / (c * (b - a))])   # vaut 1 en t = 0
    compare(f"Transitoire, E[u(1/2, {t[-1]:g})]", moy[-1], ic[-1], exacte[-1])

    ax.plot(t, exacte, "k", label="exacte")
    ax.plot(t, moy, "b--", label=f"Monte Carlo ($n = {n}$)")
    ax.fill_between(t, moy - ic, moy + ic, color="b", alpha=0.3, label="IC 95 %")
    ax.set(xlabel="$t$", ylabel="$E[u(1/2, t)]$", title=f"Transitoire, $k \\sim U({a:g}, {b:g})$")
    ax.legend()


def cas_bicouche(ax):
    """Deux couches nu_1, nu_2 iid, f = 0, u(0) = 1, u(1) = 0 : u(1/2) = nu_1 / (nu_1 + nu_2),

    donc E[u(1/2)] = 1/2 par symétrie, quelle que soit la loi.
    """
    pb = Chaleur1D(19, lambda x, nu: nu[zones(x, 2)], lambda x, nu: 0 * x, ug=1.0)   # noeud 9 en x = 1/2
    Y = monte_carlo(lambda nu: stationnaire(pb, nu)[9], tirer([stats.lognorm(s=0.5)] * 2, 1000, graine=3))
    moy, ect, ic = statistiques(Y)
    compare("Bicouche, E[u(1/2)]", moy, ic, 0.5)

    ax.hist(Y, bins=30, density=True, alpha=0.6, label="tirages de $u(1/2)$")
    ax.axvline(0.5, color="k", label="$E[u(1/2)] = 1/2$")
    ax.axvline(moy, color="r", ls="--", label="moyenne MC")
    ax.set(xlabel="$u(1/2)$", ylabel="densité", title="Bicouche, $k_1, k_2$ iid log-normales")
    ax.legend()


def cas_source(ax):
    """Source aléatoire : k = 1, f = nu pi^2 sin(pi x) avec nu ~ U(a, b), u0 = 0 : u = nu (1 - exp(-pi^2 t)) sin(pi x),

    donc E[u(1/2, t)] = E[nu] (1 - exp(-pi^2 t)).
    """
    a, b, n = 0.5, 1.5, 300
    t = np.linspace(0, 0.3, 31)
    pb = Chaleur1D(49, lambda x, nu: np.ones_like(x), lambda x, nu: nu[0] * np.pi**2 * np.sin(np.pi * x))
    Y = monte_carlo(lambda nu: transitoire(pb, nu, t)[:, 24], tirer([stats.uniform(a, b - a)], n, graine=5))
    moy, ect, ic = statistiques(Y)
    exacte = (a + b) / 2 * (1 - np.exp(-np.pi**2 * t))
    compare(f"Source, E[u(1/2, {t[-1]:g})]", moy[-1], ic[-1], exacte[-1])

    ax.plot(t, exacte, "k", label="exacte")
    ax.plot(t, moy, "b--", label=f"Monte Carlo ($n = {n}$)")
    ax.fill_between(t, moy - ic, moy + ic, color="b", alpha=0.3, label="IC 95 %")
    ax.set(xlabel="$t$", ylabel="$E[u(1/2, t)]$", title=f"Source aléatoire $\\propto U({a:g}, {b:g})$, $k = 1$")
    ax.legend()


if __name__ == "__main__":
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    print("Conductivité aléatoire (source fixée)")
    cas_stationnaire(ax[0, 0])
    cas_transitoire(ax[0, 1])
    cas_bicouche(ax[1, 0])
    print("\nSource aléatoire (conductivité fixée)")
    cas_source(ax[1, 1])

    fig.tight_layout()
    (RACINE / "resultats").mkdir(exist_ok=True)
    fig.savefig(RACINE / "resultats" / "valid_monte_carlo.png", dpi=150)
    print("\nValidation du Monte Carlo : OK")
    plt.show()
