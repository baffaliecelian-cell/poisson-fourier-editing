# Mes méthodes appliquées aux images des figures 3 à 9 de l'article, comparées aux résultats publiés.
# Les images sont extraites du PDF (environ 420 x 300 pixels, compressées), donc la comparaison
# est surtout visuelle ; les PSNR donnent un ordre de grandeur.
import os, sys
RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from methodes.solver import compute_f_hat, gradient_hat, solve_poisson
from methodes.contraste import contrast_enhance, color_from_grey, etire
from methodes.collage import colle_couleur, melange_gradient, colle_pixels, decale

DOSSIER = os.path.join(RACINE, "images", "article")
SORTIE = os.path.join(RACINE, "docs", "figures", "article")
os.makedirs(SORTIE, exist_ok=True)


# charge une image de l'article en couleur, redimensionnée à taille = (H, W) si besoin
def lit(nom, taille=None):
    im = Image.open(os.path.join(DOSSIER, nom)).convert("RGB")
    if taille is not None:                       # taille = (H, W)
        im = im.resize((taille[1], taille[0]), Image.BILINEAR)
    return np.array(im, dtype=float)


# luminance d'une image RGB (pondérations standard 0.299, 0.587, 0.114)
def gris(img):
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


# PSNR en dB entre a et b (calculé seulement dans le masque s'il est donné)
def psnr(a, b, masque=None):
    d = (np.asarray(a, float) - np.asarray(b, float)) ** 2
    if masque is not None:
        d = d[masque]
    return 10 * np.log10(255 ** 2 / d.mean())


# affiche les images côte à côte et sauvegarde la figure dans fichier
def montre(images, titres, fichier):
    fig, ax = plt.subplots(1, len(images), figsize=(4.5 * len(images), 3.6))
    for a, im, t in zip(ax, images, titres):
        if im.ndim == 2 and im.dtype == bool:
            a.imshow(im, cmap="gray")
        else:
            a.imshow(np.clip(im, 0, 255).astype(np.uint8))
        a.set_title(t, fontsize=10)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(SORTIE, fichier), dpi=110)
    plt.close()


# ---------- table 1 : Δu = Δf avec Neumann, puis même moyenne et écart-type que f ----------
print("Table 1 : reconstruction de f à partir de son gradient")
article = {"fig3": (0.00325, 71.72, 0.1136), "fig4": (0.00575, 69.33, 0.055),
           "fig5": (4.9e-5, 88.89, 0.026), "fig7": (1.15e-5, 96.46, 0.005)}
print(f"{'':6}{'MSE ici':>12}{'article':>10}{'PSNR ici':>10}{'article':>9}{'MD ici':>11}{'article':>9}")
for k, (mse_a, psnr_a, md_a) in article.items():
    f = gris(lit(k + "_1.png"))
    H, W = f.shape
    u = solve_poisson(*gradient_hat(compute_f_hat(f)), 0, H, W)
    u = (u - u.mean()) / u.std() * f.std() + f.mean()
    mse = np.mean((u - f) ** 2)
    print(f"{k:6}{mse:12.1e}{mse_a:10.1e}{10 * np.log10(255**2 / mse):10.0f}{psnr_a:9.1f}"
          f"{np.abs(u - f).max():11.1e}{md_a:9.3f}")


# ---------- figures 3 à 5 : rehaussement de contraste, T = 50, alpha = 2,5 ----------
print("\nFigures 3 à 5 : rehaussement de contraste (T = 50, alpha = 2.5)")
for k in ["fig3", "fig4", "fig5"]:
    img = lit(k + "_1.png")
    H, W = img.shape[:2]
    # masque publié (noir = sous le seuil) ; le haut blanc a pu être rogné au découpage
    m = np.array(Image.open(os.path.join(DOSSIER, k + "_2.png")).convert("L"), dtype=float)
    if m.shape[0] < H - 5:
        m = np.vstack([np.full((H - m.shape[0], m.shape[1]), 255.0), m])
    X_article = np.array(Image.fromarray(m.astype(np.uint8)).resize((W, H))) < 128
    publie = lit(k + "_3.png", (H, W))

    f = gris(img)
    for nom, fg in [("luminance", f), ("moyenne R,G,B", img.mean(axis=2))]:
        print(f"  {k} masque {nom:14}: accord avec l'article {100 * np.mean((fg < 50) == X_article):5.1f} %")
    u = contrast_enhance(f, 50, 2.5)
    res = color_from_grey(img, f, u)
    v = etire(u)                                       # min -> 0, max -> 255
    res_etire = color_from_grey(img, f, v)
    # pour comparer : le meilleur a u + b possible (moindres carrés sur le résultat publié)
    (a, b), *_ = np.linalg.lstsq(np.stack([u.ravel(), np.ones(u.size)], 1), gris(publie).ravel(), rcond=None)
    res_ideal = color_from_grey(img, f, np.clip(a * u + b, 0, 255))
    print(f"  {k} PSNR avec le résultat publié : original {psnr(img, publie):5.1f} dB, u {psnr(res, publie):5.1f} dB, "
          f"u étiré {psnr(res_etire, publie):5.1f} dB (meilleur a u + b : {psnr(res_ideal, publie):5.1f} dB)")
    print(f"       u va de {u.min():.0f} à {u.max():.0f}, facteur de l'étirement {255 / (u.max() - u.min()):.2f}")
    print(f"       niveau moyen de la zone sombre : original {f[f < 50].mean():5.1f}, publié {gris(publie)[f < 50].mean():5.1f}, "
          f"u {u[f < 50].mean():5.1f}, u étiré {v[f < 50].mean():5.1f}")
    montre([img, X_article, f < 50, publie, res, res_etire],
           ["original", "masque publié", "mon masque (luminance < 50)", "résultat publié", "u (éq. 15)",
            "u étiré (min-max)"], k + "_contraste.png")


# ---------- figure 6 : clonage du bateau (éq. 16) ----------
print("\nFigure 6 : clonage")
src, dst, publie = lit("fig6_1.png"), lit("fig6_2.png"), lit("fig6_3.png")
H, W = dst.shape[:2]
masque = np.zeros((H, W), bool)
masque[106:148, 30:284] = True                      # bateau et sillage, repérés à l'œil
# décalage (dy, dx) entre la source et le résultat publié : celui qui superpose le mieux le bateau
b = gris(src)[110:142, 190:280]
b = b - b.mean()
meilleur = None
for dy in range(-10, 80):
    for dx in range(-30, 30):
        c = gris(publie)[110 + dy:142 + dy, 190 + dx:280 + dx]
        if c.shape != b.shape:
            continue
        s = np.sum(b * (c - c.mean())) / (np.linalg.norm(b) * np.linalg.norm(c - c.mean()))
        if meilleur is None or s > meilleur[0]:
            meilleur = (s, dy, dx)
_, dy, dx = meilleur
print(f"  bateau déplacé de dy = {dy}, dx = {dx} (corrélation {meilleur[0]:.2f})")
g = decale(src, -dy, -dx)                             # g[j, l] = src[j - dy, l - dx]
M = np.roll(masque, (dy, dx), axis=(0, 1))
res = colle_couleur(dst, g, M)
print(f"  PSNR avec le résultat publié : collage des pixels {psnr(colle_pixels(dst, g, M), publie):5.1f} dB, "
      f"éq. 16 {psnr(res, publie):5.1f} dB")
montre([src, dst, publie, colle_pixels(dst, g, M), res],
       ["source", "destination", "résultat publié", "collage des pixels", "mon résultat (éq. 16)"], "fig6_clonage.png")


# ---------- figure 7 : effacer le faon (source = la même image, décalée) ----------
print("\nFigure 7 : effacement")
img = lit("fig7_1.png")
H, W = img.shape[:2]
publie = lit("fig7_2.png", (H, W))
j, l = np.mgrid[:H, :W]
masque = ((j - 162) / 34) ** 2 + ((l - 213) / 42) ** 2 < 1
res = colle_couleur(img, decale(img, 0, 90), masque)
print(f"  PSNR hors du masque : publié {psnr(img, publie, ~masque):5.1f} dB, le mien {psnr(img, res, ~masque):5.1f} dB")
montre([img, publie, res], ["original", "résultat publié", "mon résultat (éq. 16)"], "fig7_effacement.png")


# ---------- figure 8 : mélange de gradients (éq. 17), texte sur une texture ----------
print("\nFigure 8 : mélange de gradients")
texte, texture = lit("fig8_1.png"), lit("fig8_2.png")
H, W = min(texte.shape[0], texture.shape[0]), min(texte.shape[1], texture.shape[1])
texte, texture = texte[:H, :W], texture[:H, :W]
publie = lit("fig8_3.png", (H, W))
masque = np.zeros((H, W), bool)
masque[2:-2, 2:-2] = True                             # tout sauf un petit bord (pour recaler le niveau)
res = colle_couleur(texture, texte, masque, colle=melange_gradient)
seul = colle_couleur(texture, texte, masque)
print(f"  PSNR avec le résultat publié : éq. 16 {psnr(seul, publie):5.1f} dB, éq. 17 {psnr(res, publie):5.1f} dB")
montre([texte, texture, publie, seul, res],
       ["texte", "texture", "résultat publié", "éq. 16 (gradient du texte)", "mon résultat (éq. 17)"], "fig8_melange.png")


# ---------- figure 9 : texture haute fréquence collée sur une texture basse fréquence ----------
print("\nFigure 9 : texture sur texture")
f, g = lit("fig9_1.png"), lit("fig9_2.png")
publie = lit("fig9_3.png")
H, W = min(f.shape[0], g.shape[0], publie.shape[0]), min(f.shape[1], g.shape[1], publie.shape[1])
f, g, publie = f[:H, :W], g[:H, :W], publie[:H, :W]
# masque : là où le résultat publié a beaucoup de détails (écart-type local élevé)
p = gris(publie)
std = np.sqrt(np.maximum(ndimage.uniform_filter(p ** 2, 9) - ndimage.uniform_filter(p, 9) ** 2, 0))
masque = ndimage.binary_fill_holes(ndimage.binary_closing(std > 3 * np.median(std), iterations=4))
masque = ndimage.binary_opening(masque, iterations=3)
masque[:15], masque[-15:], masque[:, :15], masque[:, -15:] = False, False, False, False
res = colle_couleur(f, g, masque)
anneau = ndimage.binary_dilation(masque, iterations=10) & ~masque
print(f"  masque : {masque.sum()} pixels ; PSNR avec le publié : {psnr(res, publie):5.1f} dB, "
      f"dans le masque {psnr(res, publie, masque):5.1f} dB, juste autour {psnr(res, publie, anneau):5.1f} dB")
montre([f, g, publie, res], ["basse fréquence", "haute fréquence", "résultat publié", "mon résultat (éq. 16)"],
       "fig9_textures.png")

print("\nfigures enregistrées dans", SORTIE)
