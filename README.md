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

## Partie 2 : modèle réduit

Schéma du support : encodeur-décodeur de l'état u ↔ z_u ∈ R^{d_u}, encodeur du matériau ν → η ∈ R^{d_ν},
dynamique latente dz_u/dt = f(z_u, η). Première version, entièrement **linéaire** (référence pour les
encodeurs-décodeurs non linéaires à venir) :

| Brique | Choix linéaire |
|---|---|
| État | POD : z_u = Vᵀ(u − ū), u ≈ ū + V z_u |
| Matériau | sous-espace actif de l'équilibre en φ(ν) = log k ou s : η = Wᵀ(φ(ν) − φ̄) (une ACP de ν ne réduit rien : les ν_m sont iid) |
| Dynamique | Galerkin : f(z, η) = Vᵀ A(ν̃) V z + Vᵀ(A(ν̃) ū + b(ν̃)), ν̃ = φ⁻¹(φ̄ + Wη), intégrée exactement |

Apprentissage sur 400 matériaux, test sur 100 matériaux non vus ; l'erreur est séparée en projection de l'état,
fermeture de la dynamique latente et réduction du matériau.

Explications détaillées (motivations, méthode, tests, résultats) : [docs/reduction_lineaire.md](docs/reduction_lineaire.md).
Présentation : [presentation/reduction_lineaire.pdf](presentation/reduction_lineaire.pdf)
(figures : `python3 presentation/figures.py`, puis `pdflatex reduction_lineaire.tex` deux fois dans `presentation/`).

## Installation

    pip install -r requirements.txt

## Utilisation

    python3 main.py                          # Monte Carlo pour les deux cas (~40 s)
    python3 visualiser.py [conductivite|source]   # régime transitoire, à partir des résultats de main.py
    python3 validation/valid_solveur.py      # ordre 2 en espace sur des solutions exactes
    python3 validation/valid_monte_carlo.py  # Monte Carlo contre des espérances exactes
    python3 main_reduction.py                # modèle réduit linéaire, les deux cas (~5 s, après main.py)
    python3 validation/valid_reduction_lineaire.py   # propriétés exactes de la réduction linéaire

Les figures et les trajectoires (`.npz`, non versionnées) sont écrites dans `resultats/`.

## Structure

    chaleur/
        cas.py           les deux cas (conductivité OU source) : données, lois, variable φ(ν)
        modele.py        maillage et opérateur discret A(ν) u + b(ν)
        solveurs.py      régime transitoire (BDF) et équilibre
        monte_carlo.py   tirages, résolutions, statistiques
        affichage.py     figures
    reduction/
        lineaire.py      POD, sous-espace actif, dynamique latente de Galerkin
        affichage.py     figure de synthèse du modèle réduit
    validation/          validation du solveur, du Monte Carlo et de la réduction
    docs/                note détaillée sur le modèle réduit linéaire
    presentation/        présentation Beamer et script de ses figures
    main.py              étude Monte Carlo des deux cas
    main_reduction.py    modèle réduit linéaire des deux cas
    visualiser.py        figures du régime transitoire

## Référence

Iollo, *Réduction d'incertitude par l'équation de Liouville* (support de cours, non inclus dans le dépôt).
