"""Monte Carlo : tirages de nu (graine fixée), une résolution déterministe par tirage, statistiques."""
import numpy as np


def tirer(lois, n, graine):
    """n tirages de nu, une loi scipy.stats par composante (indépendantes) : tableau (n, len(lois))."""
    rng = np.random.default_rng(graine)
    return np.column_stack([loi.rvs(size=n, random_state=rng) for loi in lois])


def monte_carlo(resoudre, nus):
    """Applique le solveur déterministe resoudre(nu) à chaque tirage : tableau (n, ...)."""
    return np.array([resoudre(nu) for nu in nus])


def statistiques(Y):
    """Moyenne, écart-type sans biais et demi-largeur de l'IC à 95 %, sur l'axe des tirages."""
    moy = Y.mean(axis=0)
    ect = Y.std(axis=0, ddof=1)
    return moy, ect, 1.96 * ect / np.sqrt(len(Y))
