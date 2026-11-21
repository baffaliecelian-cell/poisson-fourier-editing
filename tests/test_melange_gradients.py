import os, sys
RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from methodes.collage import colle_gradient, melange_gradient, colle_couleur

def affiche(images, titres):
    fig, ax = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    for a, im, t in zip(ax, images, titres):
        a.imshow(np.clip(im, 0, 255).astype(np.uint8), cmap="gray", vmin=0, vmax=255)
        a.set_title(t)
        a.axis("off")
    plt.tight_layout()
    plt.show()

# image H x W avec un texte gris foncé sur fond clair, centré en (cj, cl)
def texte(H, W, mot, cj, cl, taille):
    im = Image.new("L", (W, H), 230)
    d = ImageDraw.Draw(im)
    police = ImageFont.truetype("arialbd.ttf", taille)
    d.text((cl, cj), mot, fill=130, font=police, anchor="mm")
    return np.array(im, dtype=float)

# rectangle [j0:j1, l0:l1]
def rectangle(H, W, j0, j1, l0, l1):
    m = np.zeros((H, W), dtype=bool)
    m[j0:j1, l0:l1] = True
    return m


# --- exemple 1 : image simple, du texte sur des rayures
H, W = 256, 384
j = np.arange(H).reshape(H, 1)
l = np.arange(W).reshape(1, W)
f = 120 + 50 * np.sin(2 * np.pi * (j + 2 * l) / 14)
g = texte(H, W, "IMA", H // 2, W // 2, 150)
m = rectangle(H, W, 50, 206, 60, 324)
affiche([g, f, colle_gradient(f, g, m), melange_gradient(f, g, m)],
        ["source g : texte", "destination f : rayures", "éq. 16 : grad g dans le rectangle",
         "éq. 17 : le plus fort des deux"])

# --- exemple 2 : un graffiti sur la roche du canyon (couleur, canal par canal)
img = np.array(Image.open(os.path.join(RACINE, "images", "canyon.jpeg")).convert("RGB"), dtype=float)
H, W = img.shape[:2]
t = texte(H, W, "POISSON", H // 2, W // 2, W // 6)
g = np.stack([t, t, t], axis=2)
m = rectangle(H, W, H // 2 - W // 8, H // 2 + W // 8, W // 12, W - W // 12)
affiche([img, colle_couleur(img, g, m), colle_couleur(img, g, m, melange_gradient)],
        ["destination f : canyon", "éq. 16 : grad g dans le rectangle", "éq. 17 : le plus fort des deux"])
