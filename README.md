# Propagation d'incertitude pour l'équation de la chaleur 1D

Équation de la chaleur 1D à données aléatoires :

    du/dt = d/dx( k(x; ν) du/dx ) + f(x; ν)   sur ]0, 1[,   u(0) = u(1) = 0,   u(x, 0) = sin(πx)

Deux cas sont traités **séparément**, avec une seule donnée aléatoire à la fois :

| Cas | Donnée aléatoire | Donnée fixée |
|---|---|---|
| Conductivité aléatoire | k = k_m sur la zone m, k_m ~ LogN(0, 0.3²) | f = 1 |
| Source aléatoire | f = s_m sur la zone m, s_m ~ U(0.5, 1.5) | k = 1 |

Discrétisation par différences finies conservatives, intégration en temps par BDF, puis propagation
de l'incertitude par Monte Carlo (une résolution déterministe par tirage de ν). Les trajectoires
sauvées servent de point de départ à la réduction par l'équation de Liouville (partie 2).

## Installation

    pip install -r requirements.txt

## Utilisation

    python3 main.py                          # Monte Carlo pour les deux cas (~40 s)
    python3 visualiser.py [conductivite|source]   # régime transitoire, à partir des résultats de main.py
    python3 validation/valid_solveur.py      # ordre 2 en espace sur des solutions exactes
    python3 validation/valid_monte_carlo.py  # Monte Carlo contre des espérances exactes

Les figures et les trajectoires (`.npz`, non versionnées) sont écrites dans `resultats/`.

## Structure

    chaleur/
        modele.py        maillage et opérateur discret A(ν) u + b(ν)
        solveurs.py      régime transitoire (BDF) et équilibre
        monte_carlo.py   tirages, résolutions, statistiques
        affichage.py     figures
    validation/          validation du solveur et du Monte Carlo
    main.py              étude Monte Carlo des deux cas
    visualiser.py        figures du régime transitoire

## Référence

Iollo, *Réduction d'incertitude par l'équation de Liouville* (support de cours, non inclus dans le dépôt).
