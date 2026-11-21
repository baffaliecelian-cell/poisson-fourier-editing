import os, sys
RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from methodes.solver3d import compute_f_hat_3d, gradient_hat_3d, solve_poisson_3d
from methodes.contraste import contrast_enhance, contrast_enhance_3d

# fantôme de tête H x W x D : air sombre, crâne clair, cerveau gris avec une texture,
# deux ventricules sombres, et dans le ventricule gauche une petite lésion à peine plus claire.
# Il est un peu flouté (sigma = 1 voxel), comme une vraie IRM, pour éviter les bords parfaits.
def fantome(H=96, W=96, D=48, bruit=2):
    j, l, k = np.mgrid[0:H, 0:W, 0:D].astype(float)
    y, x, z = (j - H / 2) / 40, (l - W / 2) / 36, (k - D / 2) / 20
    r = np.sqrt(x**2 + y**2 + z**2)
    f = np.full((H, W, D), 10.0)                                   # air
    f[r < 1] = 200                                                 # crâne
    f[r < 0.88] = 120 + 8 * np.sin(2 * np.pi * l[r < 0.88] / 12)   # cerveau
    q = (x + 0.25)**2 / 0.15**2 + y**2 / 0.3**2 + z**2 / 0.4**2
    f[(q < 1) | ((x - 0.25)**2 / 0.15**2 + y**2 / 0.3**2 + z**2 / 0.4**2 < 1)] = 30
    ventricule = q < 0.5                                           # cœur du ventricule gauche (loin du flou du bord)
    lesion = (j - H // 2)**2 + (l - (W // 2 - 9))**2 + (k - D // 2)**2 < 3**2
    f[lesion] = 38
    f = gaussian_filter(f, 1) + bruit * np.random.default_rng(0).standard_normal(f.shape)
    return f, lesion, ventricule

# trois coupes orthogonales d'un volume passant par le voxel (j0, l0, k0)
def coupes(v, j0, l0, k0):
    return [v[:, :, k0], v[j0, :, :].T, v[:, l0, :].T]

f, lesion, ventricule = fantome()
H, W, D = f.shape
j0, l0, k0 = H // 2, W // 2 - 9, D // 2     # centre de la lésion

# --- 1) reconstruction : V = grad f doit redonner f, comme en 2D
t = time.time()
f_hat = compute_f_hat_3d(f)
V1_hat, V2_hat, V3_hat = gradient_hat_3d(f_hat)
u = solve_poisson_3d(V1_hat, V2_hat, V3_hat, f_hat[0, 0, 0], H, W, D)
print(f"volume {H} x {W} x {D}, résolu en {time.time() - t:.2f} s")
print("erreur max |u - f| :", np.abs(u - f).max())
print("partie imaginaire max du gradient :", max(np.abs(np.fft.ifftn(c).imag).max() for c in (V1_hat, V2_hat, V3_hat)))

# --- 2) rehaussement des zones sombres (éq. 15) : en 3D, ou coupe par coupe en 2D
T, alpha = 50, 2.5
u3 = contrast_enhance_3d(f, T, alpha)
u2 = np.stack([contrast_enhance(f[:, :, k], T, alpha) for k in range(D)], axis=2)

# contraste lésion / ventricule, bruit, et respect de l'éq. 15 selon z (d'une coupe à la suivante) :
# on compare la différence entre deux coupes voisines à celle voulue (alpha fois celle de f dans la zone sombre)
j, l, k = np.indices(f.shape)
autour = ventricule & ((j - j0)**2 + (l - l0)**2 + (k - k0)**2 > 5**2)   # ventricule loin de la lésion
Vz = np.where(f[:, :, :-1] < T, alpha, 1) * np.diff(f, axis=2)
for nom, v in [("original", f), ("3D", u3), ("2D coupe par coupe", u2)]:
    ecart = v[lesion].mean() - v[autour].mean()
    erreur_z = np.abs(np.diff(v, axis=2) - Vz).mean()
    print(f"{nom:<20} lésion - ventricule : {ecart:5.1f}   bruit dans le ventricule : {v[autour].std():5.1f}"
          f"   écart à l'éq. 15 selon z : {erreur_z:5.2f}")

# affichage avec une fenêtre adaptée à chaque volume (comme en imagerie médicale, pas d'écrêtage à [0, 255])
fig, ax = plt.subplots(3, 3, figsize=(12, 10))
for i, (v, nom) in enumerate([(f, "original"), (u3, "éq. 15 en 3D"), (u2, "éq. 15 coupe par coupe")]):
    bas, haut = np.percentile(v, [1, 99])
    for a, c, t in zip(ax[i], coupes(v, j0, l0, k0), ["axiale", "coronale", "sagittale"]):
        a.imshow(c, cmap="gray", vmin=bas, vmax=haut)
        a.set_title(f"{nom} : coupe {t}")
        a.axis("off")
plt.tight_layout()
plt.show()
