"""Équation de la chaleur 1D à conductivité OU source aléatoire : propagation d'incertitude par Monte Carlo.

    du/dt = d/dx( k(x; nu) du/dx ) + f(x; nu)  sur ]0, 1[,   u(0) = u(1) = 0,   u(x, 0) = sin(pi x)

Deux cas traités séparément, une seule donnée aléatoire à la fois, l'autre fixée à sa valeur nominale 1 :
    - conductivité aléatoire : nu = (k_1, ..., k_M), k = k_m sur la zone m, f = 1 ;
    - source aléatoire       : nu = (s_1, ..., s_M), f = s_m sur la zone m, k = 1.

Chaque tirage nu_i donne une trajectoire u(t; nu_i) : mu_t ≈ (1/n) sum_i delta_{u(t; nu_i)} (poids w_q = 1/n).
"""
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from chaleur.cas import CAS, M
from chaleur.solveurs import transitoire, stationnaire
from chaleur.monte_carlo import tirer, monte_carlo, statistiques
from chaleur.affichage import tracer

# Paramètres (maillage N, zones M et lois : chaleur/cas.py)
n_tirages, graine = 500, 2026
t = np.linspace(0, 1, 51)                                      # instants de sortie
dossier = Path(__file__).resolve().parent / "resultats"

dossier.mkdir(exist_ok=True)
for nom, cas in CAS.items():
    titre, loi = cas.titre, cas.loi
    print(f"\n{titre}")
    pb = cas.probleme()

    # Monte Carlo : une résolution déterministe par tirage de nu
    debut = time.perf_counter()
    nus = tirer([loi] * M, n_tirages, graine)
    U = monte_carlo(lambda nu: transitoire(pb, nu, t), nus)    # trajectoires (n_tirages, len(t), N)
    U_eq = monte_carlo(lambda nu: stationnaire(pb, nu), nus)   # équilibres (n_tirages, N)
    print(f"{n_tirages} tirages en {time.perf_counter() - debut:.1f} s")

    # Observable T = g(u) : température moyenne
    T, T_eq = U.mean(axis=2), U_eq.mean(axis=1)
    moy, ect, ic = statistiques(T[:, -1])
    moy_eq, ect_eq, ic_eq = statistiques(T_eq)
    print(f"T(t_f = {t[-1]:g}) : moyenne = {moy:.4f} ± {ic:.4f} (IC 95 %), écart-type = {ect:.4f}")
    print(f"Équilibre T*  : moyenne = {moy_eq:.4f} ± {ic_eq:.4f} (IC 95 %), écart-type = {ect_eq:.4f}")

    np.savez(dossier / f"monte_carlo_{nom}.npz", x=pb.x, t=t, nu=nus, u=U)   # pour la réduction (partie 2)
    tracer(pb.x, t, U, T, T_eq, dossier / f"monte_carlo_{nom}.png", titre)
plt.show()
