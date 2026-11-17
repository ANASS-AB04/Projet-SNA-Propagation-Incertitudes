"""Figure de synthèse du modèle réduit linéaire (main_reduction.py)."""
import numpy as np
import matplotlib.pyplot as plt

from chaleur.monte_carlo import statistiques


def tracer_reduction(fichier, titre, sv, vp_acp, vp_actif, dims_u, err, t, z0, Z, Z_eq, eta, Z_vrai,
                     T_mc, T_rom, T_eq_mc, T_eq_rom, n_traj=40):
    """Spectres, erreurs selon d_u, plan latent (z_1, z_2), et statistiques de T = g(u) : Monte Carlo vs réduit.

    err : dict nom -> liste des erreurs moyennes sur le test, alignée sur dims_u.
    Z, Z_eq, eta : trajectoires (instants quelconques), équilibres et matériaux latents (d_u = d_nu = 2), alignés
    sur les mêmes matériaux ; Z_vrai : E_u(u) du modèle complet aux instants t.
    """
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))

    k = np.arange(1, min(len(sv), 20) + 1)
    ax[0, 0].semilogy(k, sv[:len(k)] / sv[0], "o-")
    ax[0, 0].set(xlabel="mode $i$", ylabel="$\\sigma_i / \\sigma_1$", title="Spectre POD des snapshots (apprentissage)")

    m = np.arange(1, len(vp_acp) + 1)
    ax[0, 1].bar(m - 0.2, vp_acp / vp_acp[0], 0.4, label="ACP de $\\varphi(\\nu)$ (covariance)")
    ax[0, 1].bar(m + 0.2, vp_actif / vp_actif[0], 0.4, label="sous-espace actif ($E[J^T J]$)")
    ax[0, 1].set(yscale="log", xticks=m, xlabel="direction", ylabel="valeur propre / la plus grande",
                 title="Réduction du matériau")
    ax[0, 1].legend()

    for nom, e in err.items():
        ax[0, 2].semilogy(dims_u, e, "o-", label=nom)
    ax[0, 2].set(xlabel="$d_u$", ylabel="erreur relative moyenne", title="Erreur sur les matériaux non vus")
    ax[0, 2].legend()

    a = ax[1, 0]
    couleurs = plt.cm.viridis((eta[:, 0] - eta[:, 0].min()) / np.ptp(eta[:, 0]))
    for i in range(min(n_traj, len(Z))):
        a.plot(Z[i, :, 0], Z[i, :, 1], color=couleurs[i], lw=0.8)
    a.scatter(Z_eq[:, 0], Z_eq[:, 1], c=eta[:, 0], s=8, cmap="viridis", zorder=3)
    for i in range(3):
        a.plot(Z_vrai[i, :, 0], Z_vrai[i, :, 1], "k--", lw=1, label="$E_u(u(t))$ : modèle complet encodé" if i == 0 else None)
    a.plot(*z0, "r*", ms=14, label="$z_0 = E_u(u_0)$")
    fig.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(eta[:, 0].min(), eta[:, 0].max())),
                 ax=a, label="$\\eta_1$")
    a.set(xlabel="$z_1$", ylabel="$z_2$", title="Plan latent : trajectoires et équilibres $z^*(\\eta)$")
    a.legend(fontsize=8)

    for T, nom, style in ((T_mc, "Monte Carlo", "-"), (T_rom, "modèle réduit", "--")):
        moy, ect, _ = statistiques(T)
        ligne, = ax[1, 1].plot(t, moy, style, label=f"{nom} : $E[T(t)]$")
        ax[1, 1].plot(t, moy + ect, style, t, moy - ect, style, color=ligne.get_color(), lw=0.6)
    ax[1, 1].set(xlabel="$t$", ylabel="$T$", title="Température moyenne : espérance ± écart-type")
    ax[1, 1].legend()

    bins = np.histogram_bin_edges(np.concatenate([T_eq_mc, T_eq_rom]), 30)
    ax[1, 2].hist(T_eq_mc, bins, density=True, alpha=0.5, label="Monte Carlo")
    ax[1, 2].hist(T_eq_rom, bins, density=True, histtype="step", lw=2, label="modèle réduit")
    ax[1, 2].set(xlabel="$T^*$", ylabel="densité", title="Loi de l'équilibre $T^*$")
    ax[1, 2].legend()

    fig.suptitle(titre)
    fig.tight_layout()
    fig.savefig(fichier, dpi=150)
