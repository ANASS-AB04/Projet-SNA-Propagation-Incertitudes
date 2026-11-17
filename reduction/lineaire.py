"""Modèle réduit à encodeur-décodeur linéaire : schéma du support d'Iollo (p. 2) avec trois briques linéaires.

    état      : z_u = E_u(u) = V^T (u - ubar),            u ≈ D_u(z_u) = ubar + V z_u             (POD)
    matériau  : eta = E_nu(nu) = W^T (phi(nu) - phibar),  nu ≈ D_nu(eta) = phi^-1(phibar + W eta)  (sous-espace actif)
    dynamique : dz_u/dt = f(z_u, eta) = Ahat z_u + chat,  Ahat = V^T A V,  chat = V^T (A ubar + b),
                avec (A, b) = operateur(D_nu(eta))                                                 (Galerkin)

A(nu) est symétrique définie négative et V orthonormée : Ahat l'est aussi, d'où un équilibre unique z*(eta), attractif,
et une trajectoire latente exacte (pas d'erreur d'intégration en temps).
"""
import copy

import numpy as np
from scipy.linalg import eigh

from chaleur.solveurs import stationnaire


class POD:
    """Encodeur-décodeur linéaire de l'état : les d premiers modes POD des snapshots centrés (dernier axe : espace)."""

    def __init__(self, snapshots, d):
        X = np.asarray(snapshots, dtype=float)
        X = X.reshape(-1, X.shape[-1])                           # (n_snapshots, N)
        self.ubar = X.mean(axis=0)
        _, self.sv, Vt = np.linalg.svd(X - self.ubar, full_matrices=False)
        self.modes = Vt.T                                        # tous les modes, par énergie décroissante
        self.V = self.modes[:, :d]                               # (N, d)
        if self.V.shape[1] != d:
            raise ValueError(f"d = {d} supérieur au nombre de modes disponibles ({self.modes.shape[1]})")

    def avec_dimension(self, d):
        """Même base POD tronquée à d modes (sans refaire la SVD)."""
        pod = copy.copy(self)
        pod.V = self.modes[:, :d]
        return pod

    def encoder(self, u):
        return (np.asarray(u) - self.ubar) @ self.V

    def decoder(self, z):
        return self.ubar + np.asarray(z) @ self.V.T


def gradient_equilibre(pb, nu, phi, phi_inv, eps=1e-4):
    """J = du*/dphi (N x M) par différences centrées en phi = phi(nu), u*(nu) solution de A(nu) u + b(nu) = 0."""
    p = phi(np.asarray(nu, dtype=float))
    J = np.empty((len(pb.x), len(p)))
    for m in range(len(p)):
        e = np.zeros_like(p)
        e[m] = eps
        J[:, m] = (stationnaire(pb, phi_inv(p + e)) - stationnaire(pb, phi_inv(p - e))) / (2 * eps)
    return J


class SousEspaceActif:
    """Encodeur-décodeur linéaire du matériau, guidé par la sortie : directions propres de C = E[J^T J], J = du*/dphi.

    Les nu_m sont iid : une ACP de phi(nu) seul ne réduirait rien (covariance isotrope). On garde ici les
    combinaisons de phi(nu) auxquelles l'équilibre u* est le plus sensible.
    """

    def __init__(self, pb, nus, d, phi, phi_inv, eps=1e-4):
        self.phi, self.phi_inv = phi, phi_inv
        nus = np.asarray(nus, dtype=float)
        self.phibar = phi(nus).mean(axis=0)
        C = np.mean([J.T @ J for J in (gradient_equilibre(pb, nu, phi, phi_inv, eps) for nu in nus)], axis=0)
        lam, Q = np.linalg.eigh(C)
        lam, Q = lam[::-1], Q[:, ::-1]                           # ordre décroissant
        Q *= np.sign(Q[np.abs(Q).argmax(axis=0), range(Q.shape[1])])   # signe : plus grande composante > 0
        self.C, self.valeurs_propres, self.directions = C, lam, Q
        self.W = Q[:, :d]                                        # (M, d)

    def avec_dimension(self, d):
        """Mêmes directions, tronquées à d (sans recalculer les gradients)."""
        mat = copy.copy(self)
        mat.W = self.directions[:, :d]
        return mat

    def encoder(self, nu):
        return (self.phi(np.asarray(nu, dtype=float)) - self.phibar) @ self.W

    def decoder(self, eta):
        return self.phi_inv(self.phibar + np.asarray(eta) @ self.W.T)


class GalerkinLineaire:
    """Dynamique latente f(z, eta) = Ahat(eta) z + chat(eta) : projection de Galerkin du modèle complet sur la POD."""

    def __init__(self, pb, pod, materiau):
        self.pb, self.pod, self.materiau = pb, pod, materiau
        self.z0 = pod.encoder(pb.u0)                             # état initial fixé : même z0 pour tout matériau

    def operateur(self, eta):
        """Ahat (d x d) et chat (d) pour le matériau latent eta."""
        A, b = self.pb.operateur(self.materiau.decoder(eta))
        V, ubar = self.pod.V, self.pod.ubar
        return V.T @ (A @ V), V.T @ (A @ ubar + b)

    def vitesse(self, z, eta):
        Ah, ch = self.operateur(eta)
        return np.asarray(z) @ Ah.T + ch

    def equilibre(self, eta):
        """z*(eta), solution de Ahat z + chat = 0."""
        Ah, ch = self.operateur(eta)
        return np.linalg.solve(Ah, -ch)

    def trajectoire(self, eta, t):
        """z(t) exacte, tableau (len(t), d) : Ahat = Q diag(lam) Q^T, z(t) = z* + Q exp(lam t) Q^T (z0 - z*)."""
        Ah, ch = self.operateur(eta)
        lam, Q = eigh(Ah)
        zs = np.linalg.solve(Ah, -ch)
        return zs + (np.exp(np.outer(t, lam)) * (Q.T @ (self.z0 - zs))) @ Q.T
