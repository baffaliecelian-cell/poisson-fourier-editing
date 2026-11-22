import os, sys
RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from methodes.collage import colle_gradient, colle_pixels, colle_couleur, decale, disque

# affiche les images côte à côte, en niveaux de gris entre 0 et 255
def affiche(images, titres):
    fig, ax = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    for a, im, t in zip(ax, images, titres):
        a.imshow(np.clip(im, 0, 255).astype(np.uint8), cmap="gray", vmin=0, vmax=255)
        a.set_title(t)
        a.axis("off")
    plt.tight_layout()
    plt.show()


# --- exemple 1 : image simple, un disque clair sur un fond qui s'éclaircit vers la droite
H, W = 256, 256
l = np.arange(W).reshape(1, W)
f = 60 + 0.5 * l + np.zeros((H, 1))
f[disque(H, W, 128, 70, 25)] += 80

# dupliquer le disque à droite : la zone (128, 190) reçoit ce qu'il y a 120 pixels à gauche
m = disque(H, W, 128, 190, 31)
g = decale(f, 0, -120)
affiche([f, colle_pixels(f, g, m), colle_gradient(f, g, m)],
        ["original", "dupliqué : copie des pixels", "dupliqué : copie du gradient"])

# effacer le disque : la zone du disque reçoit le fond situé 100 pixels à droite
m = disque(H, W, 128, 70, 32)
g = decale(f, 0, 100)
affiche([f, colle_pixels(f, g, m), colle_gradient(f, g, m)],
        ["original", "effacé : copie des pixels", "effacé : copie du gradient"])

# --- exemple 2 : photo du chat dans l'herbe
img = np.array(Image.open(os.path.join(RACINE, "images", "chat.jpg")).convert("RGB"), dtype=float)
H, W = img.shape[:2]
duplique = colle_couleur(img, decale(img, 0, 182), disque(H, W, 172, 120, 52))   # chat recopié 182 px à gauche
efface = colle_couleur(img, decale(img, -10, -160), disque(H, W, 172, 302, 52))  # herbe prise à gauche du chat
affiche([img, duplique, efface], ["original", "chat dupliqué", "chat effacé"])
