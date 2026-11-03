"""Figures des résultats Monte Carlo : synthèse (main) et régime transitoire (visualiser)."""
import numpy as np
import matplotlib.pyplot as plt

from chaleur.monte_carlo import statistiques


def tracer(x, t, U, T, T_eq, fichier, titre):
    """Profils à t_f, température moyenne E[T(t)], lois de T(t_f) et de l'équilibre T*."""
    moy, ect, _ = statistiques(U[:, -1])
    moy_T, ect_T, ic_T = statistiques(T)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    ax[0].plot(x, U[:20, -1].T, color="0.75", lw=0.8)                  # quelques tirages
    ax[0].plot(x, moy, "b", label="moyenne")
    ax[0].fill_between(x, moy - ect, moy + ect, color="b", alpha=0.2, label="± écart-type")
    ax[0].set(xlabel="$x$", ylabel="$u(x, t_f)$", title=f"Profils à $t_f = {t[-1]:g}$ (gris : tirages)")
    ax[0].legend()

    ax[1].plot(t, moy_T, "b", label="$E[T(t)]$")
    ax[1].fill_between(t, moy_T - ect_T, moy_T + ect_T, color="b", alpha=0.2, label="± écart-type")
    ax[1].fill_between(t, moy_T - ic_T, moy_T + ic_T, color="r", alpha=0.5, label="IC 95 %")
    ax[1].set(xlabel="$t$", ylabel="$T$", title="Température moyenne $T = g(u)$")
    ax[1].legend()

    ax[2].hist(T[:, -1], bins=30, density=True, alpha=0.5, label="$T(t_f)$")
    ax[2].hist(T_eq, bins=30, density=True, alpha=0.5, label="équilibre $T^*$")
    ax[2].set(xlabel="$T$", ylabel="densité", title="Loi de $T$")
    ax[2].legend()

    fig.suptitle(titre)
    fig.tight_layout()
    fig.savefig(fichier, dpi=150)


def tracer_transitoire(x, t, U, fichier, titre, instants=(0.04, 0.1, 0.2, 1.0)):
    """E[u] et écart-type de u dans le plan (x, t), profils et loi de T = g(u) à quelques instants."""
    moy, ect, _ = statistiques(U)                                      # (len(t), N)
    T = U.mean(axis=2)
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    for a, Z, nom in zip(ax[0], (moy, ect), ("$E[u(x, t)]$", "écart-type de $u(x, t)$")):
        fig.colorbar(a.pcolormesh(x, t, Z, shading="auto"), ax=a)
        a.set(xlabel="$x$", ylabel="$t$", title=nom)

    for i in [np.abs(t - ti).argmin() for ti in instants]:
        ligne, = ax[1, 0].plot(x, moy[i], label=f"$t = {t[i]:g}$")
        ax[1, 0].fill_between(x, moy[i] - ect[i], moy[i] + ect[i], color=ligne.get_color(), alpha=0.2)
        ax[1, 1].hist(T[:, i], bins=30, density=True, alpha=0.5, label=f"$t = {t[i]:g}$")
    ax[1, 0].set(xlabel="$x$", ylabel="$u$", title="Profils moyens ± écart-type")
    ax[1, 0].legend()
    ax[1, 1].set(xlabel="$T$", ylabel="densité", title="Loi de $T(t)$ au cours du temps")
    ax[1, 1].legend()

    fig.suptitle(titre)
    fig.tight_layout()
    fig.savefig(fichier, dpi=150)
