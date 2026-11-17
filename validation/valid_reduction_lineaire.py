"""Validation du modèle réduit linéaire (POD + sous-espace actif + Galerkin) sur des propriétés exactes, cas par cas.

Données générées sur place (quelques trajectoires par cas) : ne dépend pas des résultats de main.py.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.sparse.linalg import spsolve

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))                                    # rend les paquets importables
from chaleur.cas import CAS, M
from chaleur.solveurs import transitoire, stationnaire
from chaleur.monte_carlo import tirer, monte_carlo
from reduction.lineaire import POD, SousEspaceActif, GalerkinLineaire, gradient_equilibre

t = np.linspace(0, 1, 51)
n_app, n_test = 20, 20                                             # matériaux d'apprentissage et de test


def erreur_relative(u, v):
    """max sur les tirages et les instants de |u - v|_2 / |u|_2 (dernier axe : espace)."""
    return np.max(np.linalg.norm(u - v, axis=-1) / np.linalg.norm(u, axis=-1))


def test_pod(U):
    """Base orthonormée, encodeur inverse à gauche du décodeur, erreur de projection = énergie des modes écartés."""
    X = U.reshape(-1, U.shape[-1])
    d = 3
    pod = POD(X, d)
    z = np.random.default_rng(0).normal(size=(10, d))
    err_ortho = np.max(np.abs(pod.V.T @ pod.V - np.eye(d)))
    err_inv = np.max(np.abs(pod.encoder(pod.decoder(z)) - z))
    residu = np.sum((X - pod.decoder(pod.encoder(X)))**2)
    err_ey = abs(residu - np.sum(pod.sv[d:]**2)) / np.sum(pod.sv[d:]**2)
    print(f"  POD : |V^T V - I| = {err_ortho:.1e}, |E(D(z)) - z| = {err_inv:.1e}, "
          f"Eckart-Young (écart relatif) = {err_ey:.1e}")
    assert err_ortho < 1e-12 and err_inv < 1e-12, "POD : base non orthonormée ou encodeur incohérent"
    assert err_ey < 1e-10, "POD : erreur de projection différente de l'énergie des modes écartés"


def test_rang_exact(nom, pb, nus):
    """Source : u* = -A^{-1} sum_m s_m chi_m est affine en s, donc les équilibres centrés sont de rang M = 3."""
    U_eq = monte_carlo(lambda nu: stationnaire(pb, nu), nus)
    err = [erreur_relative(U_eq, POD(U_eq, d).decoder(POD(U_eq, d).encoder(U_eq))) for d in (M - 1, M)]
    print(f"  Équilibres : erreur de reconstruction POD d = {M - 1} : {err[0]:.1e}, d = {M} : {err[1]:.1e}")
    if nom == "source":
        assert err[1] < 1e-12 < 1e-3 < err[0], "source : les équilibres ne sont pas exactement de rang 3"


def gradient_analytique(pb, nu, dnu_dphi):
    """Modèle affine en nu : A(nu) = A(0) + sum nu_m (A(e_m) - A(0)), idem pour b, donc
    du*/dnu_m = -A^{-1} [ (A(e_m) - A(0)) u* + b(e_m) - b(0) ], puis multiplication par dnu_m/dphi_m."""
    A, b = pb.operateur(nu)
    u = spsolve(A, -b)
    A0, b0 = pb.operateur(np.zeros(M))
    J = np.empty((len(u), M))
    for m in range(M):
        Am, bm = pb.operateur(np.eye(M)[m])
        J[:, m] = -spsolve(A, (Am - A0) @ u + (bm - b0)) * dnu_dphi[m]
    return J


def test_materiau(nom, cas, pb, nus):
    """Gradient par différences finies = gradient analytique ; directions orthonormées ; d_nu = M : reconstruction exacte."""
    err_grad = 0.0
    for nu in nus[:5]:
        dnu_dphi = nu if nom == "conductivite" else np.ones(M)     # phi = log k : dk/dphi = k ; phi = s : 1
        J_ex = gradient_analytique(pb, nu, dnu_dphi)
        J_df = gradient_equilibre(pb, nu, cas.phi, cas.phi_inv)
        err_grad = max(err_grad, np.max(np.abs(J_df - J_ex)) / np.max(np.abs(J_ex)))
    sa = SousEspaceActif(pb, nus, M, cas.phi, cas.phi_inv)
    lam, Q = sa.valeurs_propres, sa.directions
    err_ortho = np.max(np.abs(Q.T @ Q - np.eye(M)))
    err_rec = np.max(np.abs(sa.decoder(sa.encoder(nus)) - nus) / nus)
    print(f"  Matériau : gradient DF / analytique = {err_grad:.1e}, |W^T W - I| = {err_ortho:.1e}, "
          f"reconstruction d_nu = M : {err_rec:.1e}, valeurs propres = {np.array2string(lam / lam[0], precision=3)}")
    assert err_grad < 1e-6, "gradient de l'équilibre incorrect"
    assert err_ortho < 1e-12 and np.all(lam >= -1e-14 * lam[0]) and np.all(np.diff(lam) <= 0), "directions incorrectes"
    assert err_rec < 1e-12, "d_nu = M : le matériau n'est pas reconstruit exactement"
    return sa


def test_galerkin_complet(pb, U, sa, nus_test, ax=None):
    """d_u = N et d_nu = M : V carrée orthogonale, le modèle réduit est le modèle complet (aux arrondis près)."""
    rom = GalerkinLineaire(pb, POD(U, len(pb.x)), sa.avec_dimension(M))
    err_tr = err_eq = 0.0
    for nu in nus_test[:3]:
        eta = rom.materiau.encoder(nu)
        u_ref = transitoire(pb, nu, t, rtol=1e-10, atol=1e-12)
        u_rom = rom.pod.decoder(rom.trajectoire(eta, t))
        err_tr = max(err_tr, erreur_relative(u_ref, u_rom))
        err_eq = max(err_eq, erreur_relative(stationnaire(pb, nu), rom.pod.decoder(rom.equilibre(eta))))
    print(f"  Galerkin d_u = N, d_nu = M : écart au modèle complet, transitoire = {err_tr:.1e}, équilibre = {err_eq:.1e}")
    assert err_tr < 1e-7 and err_eq < 1e-10, "le modèle réduit complet ne reproduit pas le modèle complet"
    if ax is not None:                                             # dernier matériau testé
        for i in [0, 1, 5, 50]:
            ligne, = ax.plot(pb.x, u_ref[i], lw=1, label=f"$t = {t[i]:g}$")
            ax.plot(pb.x[::3], u_rom[i, ::3], "o", mfc="none", color=ligne.get_color())


def test_integration_et_stabilite(pb, U, sa, nus_test):
    """d_u = 2 : solution exacte = formule par exponentielle de matrice, f(z*) = 0, Ahat symétrique définie négative."""
    rom = GalerkinLineaire(pb, POD(U, 2), sa.avec_dimension(2))
    err_int = err_vit = err_sym = 0.0
    lam_max = -np.inf
    for nu in nus_test:
        eta = rom.materiau.encoder(nu)
        Ah, ch = rom.operateur(eta)
        zs = rom.equilibre(eta)
        z_expm = np.array([zs + expm(Ah * ti) @ (rom.z0 - zs) for ti in t])
        err_int = max(err_int, np.max(np.abs(rom.trajectoire(eta, t) - z_expm)))
        err_vit = max(err_vit, np.max(np.abs(rom.vitesse(zs, eta))) / np.max(np.abs(ch)))
        err_sym = max(err_sym, np.max(np.abs(Ah - Ah.T)) / np.max(np.abs(Ah)))
        lam_max = max(lam_max, np.linalg.eigvalsh(Ah).max())
    print(f"  d_u = 2 : intégration exacte vs expm = {err_int:.1e}, |f(z*)| = {err_vit:.1e}, "
          f"asymétrie de Ahat = {err_sym:.1e}, plus grande valeur propre = {lam_max:.2f}")
    assert err_int < 1e-12 and err_vit < 1e-12, "trajectoire latente ou équilibre incorrects"
    assert err_sym < 1e-12 and lam_max < 0, "Ahat n'est pas symétrique définie négative"


def test_convergence(nom, pb, U, sa, nus_test, U_test, ax=None):
    """L'erreur du modèle réduit sur des matériaux non vus décroît avec d_u (d_nu = M : pas de réduction du matériau)."""
    dims = [1, 2, 4, 6]
    pod = POD(U, max(dims))
    mat = sa.avec_dimension(M)
    err_rom, err_proj = [], []
    for d in dims:
        rom = GalerkinLineaire(pb, pod.avec_dimension(d), mat)
        U_rom = np.array([rom.pod.decoder(rom.trajectoire(mat.encoder(nu), t)) for nu in nus_test])
        err_rom.append(erreur_relative(U_test, U_rom))
        err_proj.append(erreur_relative(U_test, rom.pod.decoder(rom.pod.encoder(U_test))))
    print("  Erreur max sur le test, d_u = " + ", ".join(f"{d} : {e:.1e}" for d, e in zip(dims, err_rom)))
    assert np.all(np.diff(err_rom) < 0), "l'erreur du modèle réduit ne décroît pas avec d_u"
    if ax is not None:
        ligne, = ax.semilogy(dims, err_rom, "o-", label=f"{nom} : modèle réduit")
        ax.semilogy(dims, err_proj, "s--", color=ligne.get_color(), mfc="none", label=f"{nom} : projection")


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for k, (nom, cas) in enumerate(CAS.items()):
        print(f"\n{cas.titre}")
        pb = cas.probleme()
        nus = tirer([cas.loi] * M, n_app, graine=10 + k)
        nus_test = tirer([cas.loi] * M, n_test, graine=20 + k)
        U = monte_carlo(lambda nu: transitoire(pb, nu, t), nus)
        U_test = monte_carlo(lambda nu: transitoire(pb, nu, t), nus_test)

        test_pod(U)
        test_rang_exact(nom, pb, nus)
        sa = test_materiau(nom, cas, pb, nus)
        test_galerkin_complet(pb, U, sa, nus_test, ax[0] if nom == "conductivite" else None)
        test_integration_et_stabilite(pb, U, sa, nus_test)
        test_convergence(nom, pb, U, sa, nus_test, U_test, ax[1])

    ax[0].set(xlabel="$x$", ylabel="$u$", title="Conductivité : modèle complet (traits) et réduit $d_u = N$ (cercles)")
    ax[0].legend()
    ax[1].set(xlabel="$d_u$", ylabel="erreur relative max", title="Matériaux non vus, $d_\\nu = M$")
    ax[1].legend()
    fig.tight_layout()
    (RACINE / "resultats").mkdir(exist_ok=True)
    fig.savefig(RACINE / "resultats" / "valid_reduction_lineaire.png", dpi=150)
    print("\nValidation de la réduction linéaire : OK")
    plt.show()
