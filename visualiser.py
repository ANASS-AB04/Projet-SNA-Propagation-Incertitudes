"""Régime transitoire, tracé à partir des trajectoires sauvées par main.py (sans recalcul).

    python3 visualiser.py                 # les deux cas
    python3 visualiser.py source          # un seul cas : conductivite ou source
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from chaleur.affichage import tracer_transitoire

dossier = Path(__file__).resolve().parent / "resultats"
titres = {"conductivite": "Conductivité aléatoire, source f = 1", "source": "Source aléatoire, conductivité k = 1"}

for nom in sys.argv[1:] or titres:
    d = np.load(dossier / f"monte_carlo_{nom}.npz")
    tracer_transitoire(d["x"], d["t"], d["u"], dossier / f"transitoire_{nom}.png", f"{titres[nom]} : régime transitoire")
plt.show()
