import os, sys
RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from methodes.collage import colle_gradient, colle_pixels, colle_couleur, pose, disque

# charge une image du dossier images, réduite à la largeur w si besoin
def charge(nom, w=None):
    im = Image.open(os.path.join(RACINE, "images", nom)).convert("RGB")
    if w is not None:
        im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    return np.array(im, dtype=float)

def affiche(images, titres):
    fig, ax = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    for a, im, t in zip(ax, images, titres):
        a.imshow(np.clip(im, 0, 255).astype(np.uint8), cmap="gray", vmin=0, vmax=255)
        a.set_title(t)
        a.axis("off")
    plt.tight_layout()
    plt.show()


# --- exemple 1 : image simple
# f : fond qui s'éclaircit vers la droite ; src : un damier sur fond très sombre, pris dans une autre image
H, W = 256, 256
l = np.arange(W).reshape(1, W)
f = 60 + 0.5 * l + np.zeros((H, 1))
j, i = np.mgrid[0:80, 0:80]
src = 30 + 20 * (((j // 10) + (i // 10)) % 2)
g = pose(src, H, W, 88, 140)
m = disque(H, W, 128, 180, 34)
affiche([f, colle_pixels(f, g, m), colle_gradient(f, g, m)],
        ["destination", "collage des pixels", "collage du gradient (éq. 16)"])

# --- exemple 2 : le chat (avec un peu d'herbe autour) collé sur la pelouse du jardin
f = charge("jardin.jpg")
H, W = f.shape[:2]
chat = charge("chat.jpg")[115:230, 245:360]          # carré de 115 px autour du chat
g = pose(chat, H, W, 275, 260)
m = disque(H, W, 275 + 57, 260 + 57, 52)
affiche([f, colle_pixels(f, g, m), colle_couleur(f, g, m)],
        ["destination : jardin", "collage des pixels", "collage du gradient (éq. 16)"])

# --- exemple 3 : une flaque (prise sur du bitume) collée sur de l'herbe
f = charge("herbe.jpg", 640)
H, W = f.shape[:2]
flaque = charge("flaque.jpg")[45:205, 5:389]         # la flaque avec une marge de bitume
h, w = flaque.shape[:2]
top, left = (H - h) // 2 + 60, (W - w) // 2
g = pose(flaque, H, W, top, left)
jj = np.arange(H).reshape(H, 1) - (top + h / 2)
ll = np.arange(W).reshape(1, W) - (left + w / 2)
m = (jj / (0.45 * h))**4 + (ll / (0.48 * w))**4 < 1  # rectangle arrondi autour de la flaque
affiche([f, colle_pixels(f, g, m), colle_couleur(f, g, m)],
        ["destination : herbe", "collage des pixels", "collage du gradient (éq. 16)"])
