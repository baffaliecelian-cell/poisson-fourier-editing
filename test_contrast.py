import numpy as np
import matplotlib.pyplot as plt
from utils import load_grey
from contrast import load_rgb, contrast_enhance, color_from_grey, auto_threshold

# image du dossier et choix du seuil
path = "KAZEA36IWVEMRNZ55KCRLLFPBY.jpg"
auto = True        # True : seuil auto (25 % de l'image), False : seuil fixe
T_fixe = 50
alpha = 2.5

def test_contrast(path, auto, T_fixe, alpha):
    img = load_rgb(path)
    f = load_grey(path)

    if auto:
        T = auto_threshold(f)
    else:
        T = T_fixe
    print("seuil T =", T, "(auto)" if auto else "(fixe)")

    u = contrast_enhance(f, T, alpha)
    u_color = color_from_grey(img, f, u)

    X = f < T
    print("part sombre :", X.mean())
    print("std dans X :", f[X].std(), "->", u[X].std())
    print("std hors X :", f[~X].std(), "->", u[~X].std())
    print("min, max de u :", u.min(), u.max())

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img.astype(np.uint8)); ax[0].set_title("original")
    ax[1].imshow(X, cmap="gray");       ax[1].set_title(f"zones f < {T:.0f}")
    ax[2].imshow(u_color);              ax[2].set_title(f"résultat (alpha = {alpha})")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.show()

test_contrast(path, auto, T_fixe, alpha)
