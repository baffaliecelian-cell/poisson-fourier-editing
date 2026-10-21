import os, sys
RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from methodes.contraste import contrast_enhance, color_from_grey

T = 50
alpha = 2.5
rng = np.random.default_rng(0)

# charge une image, la réduit à la largeur w, renvoie (couleur, gris)
def charge(path, w):
    im = Image.open(path).convert("RGB")
    h = round(im.height * w / im.width)
    img = np.array(im.resize((w, h), Image.LANCZOS), dtype=float)
    f = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    return img, f

# image simple : fond clair, grand carré sombre, et dedans un petit carré à peine plus clair
def carre_dans_le_sombre(bruit=0):
    f = np.full((256, 256), 170.0)
    f[48:208, 48:208] = 35
    f[98:158, 98:158] += 8                          # petit carré peu contrasté
    f += rng.normal(0, bruit, f.shape)
    return np.stack([f] * 3, axis=2), f

# applique l'éq. 15 et affiche avant / après
def montre(nom, img, f):
    X = f < T
    u = contrast_enhance(f, T, alpha)
    print(f"--- {nom} : zone sombre = {100 * X.mean():.0f} % de l'image ---")
    print(f"niveau moyen {f[X].mean():5.1f} -> {u[X].mean():5.1f} | contraste {f[X].std():5.1f} -> {u[X].std():5.1f}"
          f" | pixels sous 0 : {100 * (u[X] < 0).mean():4.1f} %")

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(np.clip(img, 0, 255).astype(np.uint8)); ax[0].set_title(f"{nom} : original")
    ax[1].imshow(color_from_grey(img, f, u));            ax[1].set_title(f"éq. 15 (T = {T}, alpha = {alpha})")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.show()

montre("carré dans le sombre", *carre_dans_le_sombre())
montre("carré dans le sombre + bruit", *carre_dans_le_sombre(bruit=3))
montre("concert à contre-jour", *charge(os.path.join(RACINE, "images", "concert.jpg"), 600))
montre("caméraman", *charge(os.path.join(RACINE, "images", "camera.png"), 512))
montre("IRM", *charge(os.path.join(RACINE, "images", "irm.tif"), 512))
