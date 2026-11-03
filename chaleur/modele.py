"""Équation de la chaleur 1D à conductivité ou source aléatoire, discrétisée en espace.

    du/dt = d/dx( k(x; nu) du/dx ) + f(x; nu)   sur ]0, L[,   u(0) = ug,  u(L) = ud,  u(x, 0) = u0(x)

Une seule des deux données dépend de nu (cas conductivité aléatoire ou cas source aléatoire).

Différences finies conservatives (conductivité aux milieux des mailles) sur N noeuds intérieurs :
    du/dt = V(u, nu) = A(nu) u + b(nu),   u dans R^N,   nu fixe en temps.
"""
import numpy as np
import scipy.sparse as sp


def zones(x, M, L=1.0):
    """Indice de la zone contenant chaque point x, parmi M zones de même longueur."""
    return np.minimum((np.asarray(x) / L * M).astype(int), M - 1)


class Chaleur1D:
    """Maillage uniforme et données ; conductivite(x, nu) et source(x, nu) renvoient des tableaux comme x."""

    def __init__(self, N, conductivite, source, u0=np.zeros_like, ug=0.0, ud=0.0, L=1.0):
        x = np.linspace(0, L, N + 2)
        self.h = x[1] - x[0]
        self.x = x[1:-1]                                   # noeuds intérieurs (inconnues)
        self.xm = 0.5 * (x[:-1] + x[1:])                   # milieux des N + 1 mailles
        self.conductivite, self.source = conductivite, source
        self.u0 = u0(self.x)
        self.ug, self.ud = ug, ud

    def operateur(self, nu):
        """Matrice creuse A(nu) et vecteur b(nu) tels que du/dt = A u + b."""
        nu = np.asarray(nu)
        k = self.conductivite(self.xm, nu)                 # conductivité de chaque maille
        h2 = self.h**2
        A = sp.diags([k[1:-1] / h2, -(k[:-1] + k[1:]) / h2, k[1:-1] / h2], [-1, 0, 1], format="csc")
        b = np.array(self.source(self.x, nu), dtype=float)
        b[0] += k[0] * self.ug / h2                        # conditions de Dirichlet
        b[-1] += k[-1] * self.ud / h2
        return A, b
