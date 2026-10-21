import os, sys
RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from methodes.utils import load_grey
from methodes.contraste import load_rgb, contrast_enhance, color_from_grey, auto_threshold

# image et paramètres
path = os.path.join(RACINE, "images", "prairie.jpg")
T_fixe = 50
alpha = 2.5

# rehausse le contraste avec le seuil T, affiche les stats et sauvegarde le résultat
def run(img, f, T, nom):
    u = contrast_enhance(f, T, alpha)
    X = f < T
    print(f"--- seuil {nom} : T = {T:.0f} ---")
    print("part sombre :", X.mean())
    print("std dans X :", f[X].std(), "->", u[X].std())
    print("std hors X :", f[~X].std(), "->", u[~X].std())
    print("min, max de u :", u.min(), u.max())
    res = color_from_grey(img, f, u)
    Image.fromarray(res).save(os.path.join(RACINE, "results", f"resultat_T{T:.0f}.png"))
    return res

def test_contrast(path):
    img = load_rgb(path)
    f = load_grey(path)

    T_auto = auto_threshold(f)
    res_fixe = run(img, f, T_fixe, "fixe")
    res_auto = run(img, f, T_auto, "auto")

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].imshow(img.astype(np.uint8)); ax[0].set_title("original")
    ax[1].imshow(res_fixe);             ax[1].set_title(f"seuil fixe T = {T_fixe}")
    ax[2].imshow(res_auto);             ax[2].set_title(f"seuil auto T = {T_auto:.0f} (25 %)")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.show()

test_contrast(path)
