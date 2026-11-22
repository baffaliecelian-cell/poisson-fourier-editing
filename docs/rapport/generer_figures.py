import os, sys, time
RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RACINE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import (gaussian_filter, rotate, uniform_filter, binary_closing, binary_opening,
                           binary_fill_holes)
from methodes.utils import grad, extend_symmetric
from methodes.solver import compute_f_hat, gradient_hat, solve_poisson
from methodes.solver3d import compute_f_hat_3d, gradient_hat_3d, solve_poisson_3d
from methodes.contraste import contrast_enhance, contrast_enhance_3d, auto_threshold, etire, color_from_grey
from methodes.collage import colle_gradient, melange_gradient, colle_pixels, colle_couleur, decale, pose, disque

# figures du rapport final : python docs/rapport/generer_figures.py
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
MED = os.path.join(RACINE, "images", "medicales")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9})

# charge une image médicale en niveaux de gris, réduite à la largeur w si besoin
def charge(nom, w=None):
    im = Image.open(os.path.join(MED, nom)).convert("L")
    if w:
        im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    return np.array(im, dtype=float)

# affiche une image dans l'axe a, sans interpolation pour voir les vrais pixels
def montre(a, im, titre, vmin=0, vmax=255, cmap="gray"):
    a.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")
    a.set_title(titre)
    a.axis("off")

# enregistre la figure courante dans le dossier des figures du rapport, puis la ferme
def sauve(nom):
    plt.savefig(os.path.join(FIG, nom), dpi=200, bbox_inches="tight")
    plt.close("all")

# gradient spectral (éq. 13) ramené sur l'image H x W
def grad_spectral(f):
    H, W = f.shape
    V1_hat, V2_hat = gradient_hat(compute_f_hat(f))
    return np.fft.ifft2(V1_hat).real[:H, :W], np.fft.ifft2(V2_hat).real[:H, :W]

# carré clair (200) sur fond sombre (50)
def carre(N=128, a=40, b=88):
    f = np.full((N, N), 50.0)
    f[a:b, a:b] = 200
    return f


# --- 1) table 1 de l'article (éq. 18) : gradient spectral et différences finies
def recale(u, f):
    return (u - u.mean()) / u.std() * f.std() + f.mean()

print("=== table 1 ===")
for nom in ["radio_thorax.jpg", "radio_main.jpg", "irm_cerveau_2.jpg", "fond_oeil.jpg"]:
    f = charge(nom, 700)
    H, W = f.shape
    t = time.time()
    f_hat = compute_f_hat(f)
    u = recale(solve_poisson(*gradient_hat(f_hat), 0, H, W), f)
    duree = time.time() - t
    fe = extend_symmetric(f)
    D1, D2 = np.roll(fe, -1, axis=0) - fe, np.roll(fe, -1, axis=1) - fe
    v = recale(solve_poisson(np.fft.fft2(D1), np.fft.fft2(D2), 0, H, W), f)
    for g, e in [("spectral", u), ("diff. finies", v)]:
        mse = np.mean((e - f)**2)
        print(f"{nom:20s} {H}x{W} {g:13s} MSE {mse:9.2e}  PSNR {10*np.log10(255**2/mse):6.1f}  MD {np.abs(e-f).max():9.2e}"
              f"  ({duree:.2f} s)")


# --- 2) Gibbs et interpolation
print("=== Gibbs ===")
fig, ax = plt.subplots(1, 4, figsize=(13, 2.8))
for a, (nom, f) in zip(ax[:2], [("bord net", carre()), ("bord flou ($\\sigma$ = 1)", gaussian_filter(carre(), 1))]):
    V1, V2 = grad_spectral(f)
    gx, gy = grad(f)
    a.plot(np.arange(25, 56), V2[64, 25:56], "o-", ms=3, label="gradient spectral")
    a.plot(np.arange(25, 56) + 0.5, gx[64, 25:56], "s--", ms=3, label="diff. finie (en k+1/2)")
    a.set_title(f"(a) dérivée horizontale, {nom}" if a is ax[0] else f"(b) dérivée horizontale, {nom}")
    a.set_xlabel("colonne")
    print(nom, "valeurs spectrales autour du bord :", np.round(V2[64, 36:44], 1))
ax[0].legend(fontsize=7, loc="lower right")
for a, order, nom in [(ax[2], 0, "(c) rotation 45°, plus proche voisin"), (ax[3], 3, "(d) rotation 45°, bicubique")]:
    r = rotate(carre(), 45, reshape=False, order=order, cval=50)
    montre(a, np.hypot(*grad_spectral(r)), nom + " : |grad|", 0, 40)
sauve("gibbs.png")


# --- 3) contraste sur une image simple : pourquoi le sombre devient noir
print("=== carré dans le sombre ===")
f = np.full((256, 256), 170.0)
f[48:208, 48:208] = 35
f[98:158, 98:158] += 8
u = contrast_enhance(f, 50, 2.5)
v = etire(u)
fond = f > 100                              # zones de l'image, pour les moyennes
petit = np.zeros_like(fond)
petit[98:158, 98:158] = True
grand = ~fond & ~petit
for nom, x in [("éq. 15", u), ("après étirement", v)]:
    print(f"{nom} : fond {x[fond].mean():.0f}, grand carré {x[grand].mean():.0f}, petit carré {x[petit].mean():.0f},"
          f" écart petit - grand {x[petit].mean() - x[grand].mean():.1f}")
print(f"u de {u.min():.0f} à {u.max():.0f}, facteur {255 / np.ptp(u):.2f}")
fig, ax = plt.subplots(1, 4, figsize=(13, 3), gridspec_kw={"width_ratios": [1, 1, 1, 1.6]})
montre(ax[0], f, "(a) original")
montre(ax[1], u, "(b) éq. 15, affiché dans [0, 255]")
montre(ax[2], v, "(c) éq. 15 puis étirement")
ax[3].plot(f[128], "k", label="f")
ax[3].plot(u[128], "b", label="u (éq. 15)")
ax[3].plot(v[128], "r--", label="u étiré")
ax[3].axhline(0, color="gray", lw=0.6)
ax[3].set_title("(d) profil de la ligne 128")
ax[3].legend(fontsize=7)
sauve("contraste_simple.png")


# --- 4) section 5 : texture haute fréquence sur texture basse fréquence (figure 9 de l'article)
print("=== figure 9 ===")
H, W = 256, 256
j = np.arange(H).reshape(H, 1)
l = np.arange(W).reshape(1, W)
f = 128 + 40 * np.sin(2 * np.pi * l / 90) * np.cos(2 * np.pi * j / 70)
g = 128 + 40 * np.cos(2 * np.pi * (j + l) / 6) + 30 * np.cos(2 * np.pi * (j - 2 * l) / 5)
m = disque(H, W, 128, 128, 60)
u = colle_gradient(f, g, m)
d = np.sqrt((j - 128)**2 + (l - 128)**2) - 60
e = np.abs(u - f)
e_max = [e[(d >= k) & (d < k + 1)].max() for k in range(60)]
print("écart max à d=0,5,10,20,40,59 :", [round(e_max[k], 1) for k in (0, 5, 10, 20, 40, 59)])
fig, ax = plt.subplots(1, 4, figsize=(13, 3))
montre(ax[0], f, "(a) f : basse fréquence")
montre(ax[1], g, "(b) g : haute fréquence")
montre(ax[2], u, "(c) collage du gradient (éq. 16)")
ax[3].semilogy(e_max)
ax[3].set_xlabel("distance au bord du masque (pixels)")
ax[3].set_title("(d) max |u - f| hors du masque")
sauve("section5.png")


# --- 5) mélange de gradients (éq. 17) sur une texture
from PIL import ImageDraw, ImageFont
H, W = 256, 384
j = np.arange(H).reshape(H, 1)
l = np.arange(W).reshape(1, W)
f = 120 + 50 * np.sin(2 * np.pi * (j + 2 * l) / 14)
im = Image.new("L", (W, H), 230)
ImageDraw.Draw(im).text((W // 2, H // 2), "IMA", fill=130, font=ImageFont.truetype("arialbd.ttf", 150), anchor="mm")
g = np.array(im, dtype=float)
m = np.zeros((H, W), dtype=bool)
m[50:206, 60:324] = True
fig, ax = plt.subplots(1, 4, figsize=(13, 2.5))
montre(ax[0], g, "(a) source g")
montre(ax[1], f, "(b) destination f")
montre(ax[2], np.clip(colle_gradient(f, g, m), 0, 255), "(c) éq. 16 dans le rectangle")
montre(ax[3], np.clip(melange_gradient(f, g, m), 0, 255), "(d) éq. 17 : le gradient le plus fort")
sauve("melange.png")


# --- 6) contraste sur une radio du thorax, seuil automatique
print("=== thorax ===")
f = charge("radio_thorax.jpg", 700)
T = auto_threshold(f)
t = time.time()
u = contrast_enhance(f, T, 2.5)
v = etire(u)
print(f"T = {T:.0f}, {f.shape}, {time.time() - t:.2f} s, u dans [{u.min():.0f}, {u.max():.0f}],"
      f" {100 * (u < 0).mean():.1f} % sous 0, facteur de l'étirement {255 / np.ptp(u):.2f}")
print(f"poumons (zone X) : moyenne f {f[f < T].mean():.0f}, u {u[f < T].mean():.0f}, u étiré {v[f < T].mean():.0f}")
fig = plt.figure(figsize=(12, 6.5))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])
montre(fig.add_subplot(gs[:, 0]), f, "(a) original")
montre(fig.add_subplot(gs[:, 1]), (f < T) * 255.0, f"(b) zone sombre X (T = {T:.0f})")
montre(fig.add_subplot(gs[:, 2]), v, "(c) éq. 15 ($\\alpha$ = 2,5) puis étirement")
montre(fig.add_subplot(gs[0, 3]), f[180:400, 90:310], "(d) zoom, original")
montre(fig.add_subplot(gs[1, 3]), v[180:400, 90:310], "(e) zoom, éq. 15 puis étirement")
sauve("thorax_contraste.png")


# --- 7) insertion de nodules dans un poumon sain (éq. 16 et 17)
f = charge("radio_thorax.jpg", 700)
s = charge("radio_thorax_sarcoidose.png", 700)
H, W = f.shape
g = pose(s[230:390, 120:280], H, W, 260, 120)
m = disque(H, W, 340, 200, 70)
zoom = (slice(240, 440), slice(100, 300))
fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
montre(ax[0], s[210:410, 100:300], "(a) source : poumon avec nodules")
montre(ax[1], colle_pixels(f, g, m)[zoom], "(b) copie des pixels")
montre(ax[2], np.clip(colle_gradient(f, g, m)[zoom], 0, 255), "(c) éq. 16")
montre(ax[3], np.clip(melange_gradient(f, g, m)[zoom], 0, 255), "(d) éq. 17")
sauve("nodules.png")


# --- 8) effacer les marqueurs « Dx » et « Sin » (source : la même image, décalée)
m1 = np.zeros((H, W), dtype=bool)
m1[20:57, 38:86] = True
m2 = np.zeros((H, W), dtype=bool)
m2[3:40, 556:616] = True
u = colle_gradient(f, decale(f, 0, 70), m1)
u = colle_gradient(u, decale(u, 0, 70), m2)
print("effacement : écart max sous la ligne 100 :", np.abs(u - f)[100:].max())
fig, ax = plt.subplots(2, 1, figsize=(10, 3.2))
montre(ax[0], f[:110], "(a) original : marqueurs « Dx » et « Sin »")
montre(ax[1], np.clip(u[:110], 0, 255), "(b) marqueurs effacés (éq. 16, source : la zone 70 pixels à droite)")
sauve("effacement.png")


# --- 9) 3D : fantôme, puis scanner de tête
print("=== 3D ===")
def fantome(H=96, W=96, D=48, bruit=2):
    j, l, k = np.mgrid[0:H, 0:W, 0:D].astype(float)
    y, x, z = (j - H / 2) / 40, (l - W / 2) / 36, (k - D / 2) / 20
    r = np.sqrt(x**2 + y**2 + z**2)
    f = np.full((H, W, D), 10.0)
    f[r < 1] = 200
    f[r < 0.88] = 120 + 8 * np.sin(2 * np.pi * l[r < 0.88] / 12)
    q = (x + 0.25)**2 / 0.15**2 + y**2 / 0.3**2 + z**2 / 0.4**2
    f[(q < 1) | ((x - 0.25)**2 / 0.15**2 + y**2 / 0.3**2 + z**2 / 0.4**2 < 1)] = 30
    f[(j - H // 2)**2 + (l - (W // 2 - 9))**2 + (k - D // 2)**2 < 3**2] = 38
    return gaussian_filter(f, 1) + bruit * np.random.default_rng(0).standard_normal(f.shape)

f = fantome()
H, W, D = f.shape
f_hat = compute_f_hat_3d(f)
print("fantôme, reconstruction :", np.abs(solve_poisson_3d(*gradient_hat_3d(f_hat), f_hat[0, 0, 0], H, W, D) - f).max())
u3 = contrast_enhance_3d(f, 50, 2.5)
u2 = np.stack([contrast_enhance(f[:, :, k], 50, 2.5) for k in range(D)], axis=2)
fig, ax = plt.subplots(3, 3, figsize=(9, 7.5))
for i, (v, nom) in enumerate([(f, "original"), (etire(u3), "éq. 15 en 3D, étiré"), (etire(u2), "coupe par coupe, étiré")]):
    for a, c, t in zip(ax[i], [v[:, :, D // 2], v[H // 2].T, v[:, W // 2 - 9].T], ["axiale", "coronale", "sagittale"]):
        montre(a, c, f"{nom}, coupe {t}")
sauve("fantome_3d.png")

im = Image.open(os.path.join(MED, "scanner_tete_3d.tif"))
v = []
for k in range(im.n_frames):
    im.seek(k)
    v.append(np.array(im, dtype=float))
v = np.transpose(np.array(v), (1, 2, 0))
v = v / np.percentile(v, 99.5) * 255
t = time.time()
u3 = contrast_enhance_3d(v, 60, 2.5)
e3 = etire(u3)
print(f"scanner {v.shape} : {time.time() - t:.2f} s, cerveau (médiane) {np.median(v[90:200, 70:190, 5]):.0f}"
      f" -> {np.median(u3[90:200, 70:190, 5]):.0f}, étiré {np.median(e3[90:200, 70:190, 5]):.0f}"
      f" | u3 de {u3.min():.0f} à {u3.max():.0f}, facteur {255 / np.ptp(u3):.2f}")
fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
montre(ax[0], v[:, :, 5], "(a) coupe 5, original")
montre(ax[1], e3[:, :, 5], "(b) éq. 15 en 3D (T = 60), étiré")
montre(ax[2], v[90:200, 70:190, 5], "(c) zoom, original")
montre(ax[3], e3[90:200, 70:190, 5], "(d) zoom, éq. 15 en 3D, étiré")
sauve("scanner_3d.png")


# --- 10) IRM 3D du cerveau (OpenNeuro ds000001) : coupes, vue 3D, éq. 15 en 3D et coupe par coupe
import nibabel as nib
print("=== IRM 3D ===")
im = nib.load(os.path.join(MED, "irm_3d", "irm_cerveau_3d_T1.nii.gz"))
v = np.asanyarray(im.dataobj).astype(float)
v = v / np.percentile(v, 99.5) * 255
z = np.array(im.header.get_zooms(), dtype=float)       # taille d'un voxel (mm), axes R, A, S
print("IRM", v.shape, z)

# les trois coupes d'un volume (axes R, A, S), tournées pour avoir le haut de la tête en haut
def coupes_irm(v, i, j, k):
    return [np.rot90(v[i]), np.rot90(v[:, j]), np.rot90(v[:, :, k])]

# vue 3D : les trois coupes orthogonales placées dans l'espace
def plans_3d(a, v, z, i, j, k, bas, haut):
    X, Y, Z = [np.arange(n) * s for n, s in zip(v.shape, z)]
    gris = lambda c: plt.cm.gray(np.clip((c - bas) / (haut - bas), 0, 1))
    yy, zz = np.meshgrid(Y, Z, indexing="ij")
    a.plot_surface(np.full_like(yy, X[i]), yy, zz, facecolors=gris(v[i]), rstride=1, cstride=1, shade=False)
    xx, zz = np.meshgrid(X, Z, indexing="ij")
    a.plot_surface(xx, np.full_like(xx, Y[j]), zz, facecolors=gris(v[:, j]), rstride=1, cstride=1, shade=False)
    xx, yy = np.meshgrid(X, Y, indexing="ij")
    a.plot_surface(xx, yy, np.full_like(xx, Z[k]), facecolors=gris(v[:, :, k]), rstride=1, cstride=1, shade=False)
    a.set_box_aspect((X[-1], Y[-1], Z[-1]))
    a.view_init(elev=20, azim=-50)
    a.set_axis_off()

i0, j0, k0 = 75, 100, 110
fig = plt.figure(figsize=(13, 3.6))
for n, (c, t, asp) in enumerate(zip(coupes_irm(v, i0, j0, k0), ["sagittale", "coronale", "axiale"],
                                    [z[2] / z[1], z[2] / z[0], z[1] / z[0]])):
    a = fig.add_subplot(1, 4, n + 1)
    a.imshow(c, cmap="gray", vmin=0, vmax=255, aspect=asp)
    a.set_title(f"({'abc'[n]}) coupe {t}")
    a.axis("off")
a = fig.add_subplot(1, 4, 4, projection="3d")
plans_3d(a, v[::2, ::2, ::2], 2 * z, i0 // 2, j0 // 2, k0 // 2, 0, 255)
a.set_title("(d) vue 3D des trois coupes")
sauve("irm_3d.png")

# éq. 15 sur le volume réduit d'un facteur 2 (le volume prolongé complet ferait 47 millions de voxels)
w = v.reshape(80, 2, 96, 2, 96, 2).mean(axis=(1, 3, 5))
T, alpha = 60, 2.5
t = time.time()
u3 = contrast_enhance_3d(w, T, alpha)
duree = time.time() - t
u2 = np.stack([contrast_enhance(w[:, :, k], T, alpha) for k in range(w.shape[2])], axis=2)
Vz = np.where(w[:, :, :-1] < T, alpha, 1) * np.diff(w, axis=2)
print(f"volume réduit {w.shape}, part sous T : {(w < T).mean():.2f}, éq. 15 en 3D : {duree:.1f} s")
for nom, x in [("original", w), ("3D", u3), ("coupe par coupe", u2)]:
    print(f"{nom:16s} écart à l'éq. 15 selon z : {np.abs(np.diff(x, axis=2) - Vz).mean():.2f}")
print(f"u3 de {u3.min():.0f} à {u3.max():.0f} (facteur {255 / np.ptp(u3):.2f}),"
      f" u2 de {u2.min():.0f} à {u2.max():.0f} (facteur {255 / np.ptp(u2):.2f})")
fig, ax = plt.subplots(3, 3, figsize=(9, 9.5))
for r, (x, nom) in enumerate([(w, "original"), (etire(u3), "éq. 15 en 3D, étiré"), (etire(u2), "coupe par coupe, étiré")]):
    for a, c, t, asp in zip(ax[r], coupes_irm(x, 37, 50, 55), ["sagittale", "coronale", "axiale"],
                            [z[2] / z[1], z[2] / z[0], z[1] / z[0]]):
        a.imshow(c, cmap="gray", vmin=0, vmax=255, aspect=asp)
        a.set_title(f"{nom}, {t}")
        a.axis("off")
sauve("irm_contraste_3d.png")


# --- 11) comparaison avec les figures de l'article (panneaux extraits du PDF, environ 420 x 300 pixels)
print("=== figures de l'article ===")
ART = os.path.join(RACINE, "images", "article")

# charge une figure de l'article en couleur, redimensionnée à taille = (H, W) si besoin
def lit(nom, taille=None):
    im = Image.open(os.path.join(ART, nom)).convert("RGB")
    if taille is not None:
        im = im.resize((taille[1], taille[0]), Image.BILINEAR)
    return np.array(im, dtype=float)

# luminance d'une image RGB (pondérations standard 0.299, 0.587, 0.114)
def lum(img):
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

# affiche une image couleur dans l'axe a
def couleur(a, img, titre):
    a.imshow(np.clip(img, 0, 255).astype(np.uint8))
    a.set_title(titre)
    a.axis("off")

# table 1 sur les images de l'article
for k in ["fig3", "fig4", "fig5", "fig7"]:
    f = lum(lit(k + "_1.png"))
    H, W = f.shape
    u = recale(solve_poisson(*gradient_hat(compute_f_hat(f)), 0, H, W), f)
    mse = np.mean((u - f)**2)
    print(f"{k} : MSE {mse:.1e}, PSNR {10 * np.log10(255**2 / mse):.0f}, MD {np.abs(u - f).max():.1e}")

# figures 3 à 5 : rehaussement de contraste
fig, ax = plt.subplots(3, 4, figsize=(12, 6.6))
for i, k in enumerate(["fig3", "fig4", "fig5"]):
    img = lit(k + "_1.png")
    H, W = img.shape[:2]
    publie = lit(k + "_3.png", (H, W))
    f = lum(img)
    u = contrast_enhance(f, 50, 2.5)
    v = etire(u)
    print(f"{k} : u de {u.min():.0f} à {u.max():.0f}, facteur {255 / np.ptp(u):.2f} ; zone sombre : f {f[f < 50].mean():.0f},"
          f" article {lum(publie)[f < 50].mean():.0f}, u {u[f < 50].mean():.0f}, étiré {v[f < 50].mean():.0f}")
    titres = [f"fig. {k[3]}, original", "article", "éq. 15", "éq. 15 puis étirement"]
    for c, (a, im) in enumerate(zip(ax[i], [img, publie, color_from_grey(img, f, u), color_from_grey(img, f, v)])):
        couleur(a, im, titres[c] if (i == 0 or c == 0) else "")
sauve("article_contraste.png")

# zoom sur les taches de couleur (berge du lac, figure 5)
img = lit("fig5_1.png")
f = lum(img)
v = etire(contrast_enhance(f, 50, 2.5))
zone = (slice(170, 282), slice(0, 260))
fig, ax = plt.subplots(1, 2, figsize=(8, 2.4))
for a, im, t in zip(ax, [img, color_from_grey(img, f, v)],
                    ["(a) original", "(b) étirement, recoloration u / f"]):
    couleur(a, im[zone], t)
sauve("article_taches.png")

# figures 6 à 9 : collage, effacement, mélange
src, dst, pub6 = lit("fig6_1.png"), lit("fig6_2.png"), lit("fig6_3.png")
H, W = dst.shape[:2]
m = np.zeros((H, W), dtype=bool)
m[146:188, 30:284] = True                   # bateau et sillage, descendus de 40 pixels comme dans l'article
res6 = colle_couleur(dst, decale(src, -40, 0), m)

img7 = lit("fig7_1.png")
H, W = img7.shape[:2]
pub7 = lit("fig7_2.png", (H, W))
j, l = np.mgrid[:H, :W]
res7 = colle_couleur(img7, decale(img7, 0, 90), ((j - 162) / 34)**2 + ((l - 213) / 42)**2 < 1)

texte, bois = lit("fig8_1.png"), lit("fig8_2.png")
H, W = min(texte.shape[0], bois.shape[0]), min(texte.shape[1], bois.shape[1])
texte, bois = texte[:H, :W], bois[:H, :W]
pub8 = lit("fig8_3.png", (H, W))
m = np.zeros((H, W), dtype=bool)
m[2:-2, 2:-2] = True
res8 = colle_couleur(bois, texte, m, colle=melange_gradient)

f9, g9, pub9 = lit("fig9_1.png"), lit("fig9_2.png"), lit("fig9_3.png")
H, W = min(f9.shape[0], g9.shape[0], pub9.shape[0]), min(f9.shape[1], g9.shape[1], pub9.shape[1])
f9, g9, pub9 = f9[:H, :W], g9[:H, :W], pub9[:H, :W]
p = lum(pub9)                               # masque : là où le résultat publié a beaucoup de détails
std = np.sqrt(np.maximum(uniform_filter(p**2, 9) - uniform_filter(p, 9)**2, 0))
m = binary_opening(binary_fill_holes(binary_closing(std > 3 * np.median(std), iterations=4)), iterations=3)
m[:15], m[-15:], m[:, :15], m[:, -15:] = False, False, False, False
res9 = colle_couleur(f9, g9, m)

fig, ax = plt.subplots(2, 4, figsize=(15, 5.2))
for a, im, t in zip(ax[0], [pub6, pub7, pub8, pub9], ["(a) article, fig. 6", "(b) article, fig. 7",
                                                     "(c) article, fig. 8", "(d) article, fig. 9"]):
    couleur(a, im, t)
for a, im, t in zip(ax[1], [res6, res7, res8, res9], ["(e) mon code, éq. 16", "(f) mon code, éq. 16",
                                                     "(g) mon code, éq. 17", "(h) mon code, éq. 16"]):
    couleur(a, im, t)
sauve("article_collage.png")
