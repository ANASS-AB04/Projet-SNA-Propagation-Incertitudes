"""Les deux cas étudiés, traités séparément : une seule donnée aléatoire à la fois, l'autre fixée à 1.

    - conductivité aléatoire : nu = (k_1, ..., k_M), k = k_m sur la zone m, f = 1 ;
    - source aléatoire       : nu = (s_1, ..., s_M), f = s_m sur la zone m, k = 1.

Définitions partagées par le Monte Carlo (main.py) et la réduction (main_reduction.py).
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats

from chaleur.modele import Chaleur1D, zones

N, M = 50, 3                                                   # noeuds intérieurs, zones


def par_zone(x, nu):
    """nu_m sur la zone m : donnée aléatoire."""
    return nu[zones(x, M)]


def nominale(x, nu):
    """1 partout, indépendante de nu : donnée déterministe."""
    return np.ones_like(x)


def identite(nu):
    return nu


@dataclass(frozen=True)
class Cas:
    """Un cas d'étude. phi envoie nu dans la variable où l'encodeur du matériau est linéaire, phi_inv en revient."""
    titre: str
    conductivite: Callable
    source: Callable
    loi: object                                                # loi scipy.stats de chaque nu_m (iid)
    phi: Callable
    phi_inv: Callable

    def probleme(self, N=N):
        return Chaleur1D(N, self.conductivite, self.source, u0=lambda x: np.sin(np.pi * x))


CAS = {
    "conductivite": Cas("Conductivité aléatoire k_m ~ LogN(0, 0.3²), source f = 1",
                        par_zone, nominale, stats.lognorm(s=0.3), np.log, np.exp),     # log k_m ~ N(0, 0.3²)
    "source": Cas("Source aléatoire s_m ~ U(0.5, 1.5), conductivité k = 1",
                  nominale, par_zone, stats.uniform(0.5, 1.0), identite, identite),    # moyenne 1
}
