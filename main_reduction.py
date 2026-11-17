"""Partie 2, étape 1 : modèle réduit à encodeur-décodeur linéaire, cas par cas (conductivité OU source).

Schéma du support d'Iollo (p. 2), version linéaire (reduction/lineaire.py) :
    état      : z_u = E_u(u) = V^T (u - ubar),            u ≈ D_u(z_u) = ubar + V z_u     (POD)
    matériau  : eta = E_nu(nu) = W^T (phi(nu) - phibar)                                   (sous-espace actif)
    dynamique : dz_u/dt = f(z_u, eta) = Ahat(eta) z_u + chat(eta)                         (Galerkin, exacte en temps)

Les 400 premiers matériaux de main.py servent à l'apprentissage, les 100 suivants au test (matériaux non vus).
Trois erreurs sont séparées : projection de l'état, fermeture de la dynamique latente, réduction du matériau.
Prérequis : python3 main.py (trajectoires resultats/monte_carlo_*.npz).
"""
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from chaleur.cas import CAS, M
from chaleur.solveurs import transitoire, stationnaire
from chaleur.monte_carlo import monte_carlo, statistiques
from reduction.lineaire import POD, SousEspaceActif, GalerkinLineaire
from reduction.affichage import tracer_reduction

# Paramètres
n_app = 400                                                    # matériaux d'apprentissage, les suivants : test
d_u, d_nu = 2, 2                                               # modèle retenu (Liouville en 2D, support p. 2)
dims_u, dims_nu = [1, 2, 3, 4, 6], [2, 3]                      # étude des dimensions
dossier = Path(__file__).resolve().parent / "resultats"


def erreurs(u, v):
    """Erreur relative |u - v|_2 / |u|_2 par tirage et par instant : (moyenne, max)."""
    e = np.linalg.norm(u - v, axis=-1) / np.linalg.norm(u, axis=-1)
    return e.mean(), e.max()


def resoudre(rom, nus, t):
    """Matériaux latents, trajectoires et équilibres latents pour chaque nu."""
    eta = rom.materiau.encoder(nus)
    return eta, np.array([rom.trajectoire(e, t) for e in eta]), np.array([rom.equilibre(e) for e in eta])


for nom, cas in CAS.items():
    print(f"\n{'=' * 100}\n{cas.titre}\n{'=' * 100}")
    d = np.load(dossier / f"monte_carlo_{nom}.npz")
    t, nus, U = d["t"], d["nu"], d["u"]
    pb = cas.probleme()
    assert np.allclose(pb.x, d["x"]), "maillage différent de celui de main.py"
    app, test = slice(None, n_app), slice(n_app, None)
    U_eq = monte_carlo(lambda nu: stationnaire(pb, nu), nus)

    # Encodeurs-décodeurs appris sur les matériaux d'apprentissage
    debut = time.perf_counter()
    pod = POD(U[app], max(dims_u))
    sa = SousEspaceActif(pb, nus[app], M, cas.phi, cas.phi_inv)
    print(f"Apprentissage (POD + sous-espace actif) : {time.perf_counter() - debut:.1f} s")
    vp_acp = np.linalg.eigvalsh(np.cov(cas.phi(nus[app]).T))[::-1]
    print(f"POD : sigma_i / sigma_1 = {np.array2string(pod.sv[:7] / pod.sv[0], precision=1)}")
    print(f"ACP de phi(nu)    : valeurs propres / max = {np.array2string(vp_acp / vp_acp[0], precision=2)}  "
          "(isotrope : aucune réduction)")
    print(f"Sous-espace actif : valeurs propres / max = {np.array2string(sa.valeurs_propres / sa.valeurs_propres[0], precision=3)}")
    for j in range(M):
        print(f"    direction {j + 1} : {np.array2string(sa.directions[:, j], precision=2)}")
    for dn in dims_nu:
        mat = sa.avec_dimension(dn)
        e_nu = np.linalg.norm(nus[test] - mat.decoder(mat.encoder(nus[test])), axis=1) / np.linalg.norm(nus[test], axis=1)
        print(f"Reconstruction du matériau, d_nu = {dn} : erreur relative moyenne = {e_nu.mean():.1e}")

    # Étude des dimensions sur les matériaux non vus
    print(f"\n{'d_u':>4} {'d_nu':>5} | {'projection moy / max':>21} | {'modèle réduit moy / max':>24} | "
          f"{'équilibre moy / max':>20}")
    err = {"projection": [], **{f"modèle réduit, $d_\\nu = {dn}$": [] for dn in dims_nu},
           f"équilibre, $d_\\nu = {d_nu}$": []}
    tableau = np.empty((len(dims_nu), len(dims_u), 3, 2))         # (d_nu, d_u, projection/réduit/équilibre, moy/max)
    for a, dn in enumerate(dims_nu):
        for b, du in enumerate(dims_u):
            rom = GalerkinLineaire(pb, pod.avec_dimension(du), sa.avec_dimension(dn))
            _, Z, Z_eq = resoudre(rom, nus[test], t)
            e_proj = erreurs(U[test], rom.pod.decoder(rom.pod.encoder(U[test])))
            e_rom = erreurs(U[test], rom.pod.decoder(Z))
            e_eq = erreurs(U_eq[test], rom.pod.decoder(Z_eq))
            tableau[a, b] = e_proj, e_rom, e_eq
            print(f"{du:>4} {dn:>5} | {e_proj[0]:>10.1e} / {e_proj[1]:.1e} | {e_rom[0]:>13.1e} / {e_rom[1]:.1e} | "
                  f"{e_eq[0]:>9.1e} / {e_eq[1]:.1e}")
            err[f"modèle réduit, $d_\\nu = {dn}$"].append(e_rom[0])
            if dn == dims_nu[0]:
                err["projection"].append(e_proj[0])
            if dn == d_nu:
                err[f"équilibre, $d_\\nu = {d_nu}$"].append(e_eq[0])

    # Modèle retenu : statistiques de T = g(u) sur les 500 matériaux, appariées avec le Monte Carlo
    rom = GalerkinLineaire(pb, pod.avec_dimension(d_u), sa.avec_dimension(d_nu))
    eta, Z, Z_eq = resoudre(rom, nus, t)
    U_rom, U_eq_rom = rom.pod.decoder(Z), rom.pod.decoder(Z_eq)
    T_mc, T_rom, T_eq_mc, T_eq_rom = U.mean(axis=2), U_rom.mean(axis=2), U_eq.mean(axis=1), U_eq_rom.mean(axis=1)
    print(f"\nModèle retenu d_u = {d_u}, d_nu = {d_nu}, statistiques sur les {len(nus)} matériaux "
          f"(dont {n_app} d'apprentissage) :")
    moy_mc, ect_mc, ic_mc = statistiques(T_mc)
    moy_rom, ect_rom, _ = statistiques(T_rom)
    print(f"  t = 0 : u0 commun à tous les matériaux (IC nul), |D_u(E_u(u0)) - u0| / |u0| = "
          f"{np.linalg.norm(rom.pod.decoder(rom.z0) - pb.u0) / np.linalg.norm(pb.u0):.1e}")
    i = 1 + np.argmax(np.abs(moy_rom - moy_mc)[1:])                # t > 0
    print(f"  E[T(t)], t > 0 : plus grand écart en t = {t[i]:g} : |ROM - MC| = {abs(moy_rom[i] - moy_mc[i]):.1e}, "
          f"demi-largeur de l'IC 95 % du MC = {ic_mc[i]:.1e}")
    for nom_T, a, b in (("T(t_f)", T_mc[:, -1], T_rom[:, -1]), ("T*    ", T_eq_mc, T_eq_rom)):
        (ma, ea, ia), (mb, eb, _) = statistiques(a), statistiques(b)
        print(f"  {nom_T} : MC moyenne = {ma:.4f} ± {ia:.4f}, écart-type = {ea:.4f} | "
              f"ROM moyenne = {mb:.4f}, écart-type = {eb:.4f} | écart des moyennes / IC = {abs(mb - ma) / ia:.2f}")

    # Coût en ligne par matériau
    debut = time.perf_counter()
    for nu in nus[test][:20]:
        transitoire(pb, nu, t)
    t_fom = (time.perf_counter() - debut) / 20
    debut = time.perf_counter()
    for nu in nus[test][:20]:
        rom.pod.decoder(rom.trajectoire(rom.materiau.encoder(nu), t))
    t_rom = (time.perf_counter() - debut) / 20
    print(f"  Coût par matériau : modèle complet {1e3 * t_fom:.1f} ms, modèle réduit {1e3 * t_rom:.2f} ms "
          f"(x {t_fom / t_rom:.0f})")

    # Sauvegardes : figure et état réduit (point de départ de la Liouville, partie 3)
    t_fin = np.concatenate([[0], np.geomspace(1e-4, t[-1], 300)])     # trajectoire latente exacte à tout instant
    Z_fin = np.array([rom.trajectoire(e, t_fin) for e in eta[test][:40]])
    tracer_reduction(dossier / f"reduction_lineaire_{nom}.png",
                     f"{cas.titre} : modèle réduit linéaire ($d_u = {d_u}$, $d_\\nu = {d_nu}$)",
                     pod.sv, vp_acp, sa.valeurs_propres, dims_u, err, t, rom.z0, Z_fin, Z_eq[test], eta[test],
                     rom.pod.encoder(U[test]), T_mc, T_rom, T_eq_mc, T_eq_rom)
    np.savez(dossier / f"reduction_lineaire_{nom}.npz", x=pb.x, t=t, nu=nus, n_app=n_app,
             ubar=rom.pod.ubar, V=rom.pod.V, phibar=sa.phibar, W=rom.materiau.W,
             eta=eta, z0=rom.z0, z=Z, z_eq=Z_eq, t_fin=t_fin, z_fin=Z_fin, T_eq_mc=T_eq_mc,
             sv=pod.sv, vp_acp=vp_acp, vp_actif=sa.valeurs_propres, directions=sa.directions,
             dims_u=dims_u, dims_nu=dims_nu, tableau=tableau, cout=[t_fom, t_rom])
plt.show()
