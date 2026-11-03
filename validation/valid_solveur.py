"""Validation du solveur déterministe sur des solutions exactes (ordre 2 en espace attendu)."""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))                                    # rend le paquet chaleur importable
from chaleur.modele import Chaleur1D, zones
from chaleur.solveurs import transitoire, stationnaire

Ns = [10, 20, 40, 80, 160]                                         # nombres de noeuds intérieurs


def convergence(nom, h, err, ax):
    """Affiche erreurs et ordres locaux, vérifie l'ordre 2 (pente log-log) et trace l'erreur selon h."""
    print(f"\n{nom}\n{'N':>5} {'h':>9} {'erreur':>9} {'ordre':>6}")
    for i in range(len(Ns)):
        o = f"{np.log(err[i - 1] / err[i]) / np.log(h[i - 1] / h[i]):.3f}" if i > 0 else "-"
        print(f"{Ns[i]:>5} {h[i]:>9.2e} {err[i]:>9.2e} {o:>6}")
    p = np.polyfit(np.log(h), np.log(err), 1)[0]
    print(f"ordre global (pente log-log) = {p:.3f}")
    assert abs(p - 2) < 0.1, "ordre de convergence différent de 2"
    ax.loglog(h, err, "o-", label=f"{nom} (ordre {p:.2f})")


def cas_transitoire(ax):
    """k = 0.8, f = 0, u0 = sin(pi x), u(0) = u(1) = 0 : u(x, t) = exp(-k pi^2 t) sin(pi x)."""
    k, tf = 0.8, 0.1
    h, err = [], []
    for N in Ns:
        pb = Chaleur1D(N, lambda x, nu: np.full_like(x, nu[0]), lambda x, nu: 0 * x,
                       u0=lambda x: np.sin(np.pi * x))
        u = transitoire(pb, [k], [0, tf], rtol=1e-10, atol=1e-12)[-1]    # erreur en temps négligeable
        h.append(pb.h)
        err.append(np.max(np.abs(u - np.exp(-k * np.pi**2 * tf) * np.sin(np.pi * pb.x))))
    convergence("Transitoire", np.array(h), np.array(err), ax)


def cas_manufacture(ax):
    """Stationnaire, k = 1 + x et u = sin(pi x) : f = -(k u')' = (1 + x) pi^2 sin(pi x) - pi cos(pi x)."""
    h, err = [], []
    for N in Ns:
        pb = Chaleur1D(N, lambda x, nu: 1 + x,
                       lambda x, nu: (1 + x) * np.pi**2 * np.sin(np.pi * x) - np.pi * np.cos(np.pi * x))
        h.append(pb.h)
        err.append(np.max(np.abs(stationnaire(pb, []) - np.sin(np.pi * pb.x))))   # nu vide : déterministe
    convergence("Manufacturé", np.array(h), np.array(err), ax)


def cas_bicouche(ax):
    """Deux couches k1 = 1, k2 = 5, f = 0, u(0) = 1, u(1) = 0 : u affine par morceaux (flux q constant).

    Interface sur un noeud : le schéma est exact ; le transitoire depuis u0 = 0 doit tendre vers l'équilibre.
    """
    k1, k2 = 1.0, 5.0
    q = 1 / (0.5 / k1 + 0.5 / k2)                                  # résistances thermiques en série

    def exacte(x):
        return np.where(x <= 0.5, 1 - q * x / k1, q * (1 - x) / k2)

    pb = Chaleur1D(19, lambda x, nu: nu[zones(x, 2)], lambda x, nu: 0 * x, ug=1.0)   # h = 1/20 : noeud en 1/2
    u_eq = stationnaire(pb, [k1, k2])
    u_tr = transitoire(pb, [k1, k2], [0, 5], rtol=1e-10, atol=1e-12)[-1]
    err_eq, err_tr = np.max(np.abs(u_eq - exacte(pb.x))), np.max(np.abs(u_tr - exacte(pb.x)))
    print(f"\nBicouche : erreur équilibre = {err_eq:.1e}, erreur transitoire à t = 5 = {err_tr:.1e}")
    assert err_eq < 1e-12 and err_tr < 1e-6, "bicouche : solution exacte non retrouvée"

    xf = np.linspace(0, 1, 201)
    ax.plot(xf, exacte(xf), "k", label="exacte")
    ax.plot(pb.x, u_eq, "o", mfc="none", label="stationnaire")
    ax.plot(pb.x, u_tr, "+", ms=10, label="transitoire à $t = 5$")
    ax.set(xlabel="$x$", ylabel="$u$", title=f"Bicouche $k_1 = {k1:g}$, $k_2 = {k2:g}$")
    ax.legend()


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    cas_transitoire(ax[0])
    cas_manufacture(ax[0])
    h = 1 / (np.array(Ns) + 1)
    ax[0].loglog(h, 0.1 * h**2, "k--", label="$h^2$")
    ax[0].set(xlabel="$h$", ylabel="erreur max", title="Convergence en espace")
    ax[0].legend()
    cas_bicouche(ax[1])

    fig.tight_layout()
    (RACINE / "resultats").mkdir(exist_ok=True)
    fig.savefig(RACINE / "resultats" / "valid_solveur.png", dpi=150)
    print("\nValidation du solveur : OK")
    plt.show()
