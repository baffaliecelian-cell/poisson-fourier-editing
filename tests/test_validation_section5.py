import os, sys
RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

import numpy as np
import matplotlib.pyplot as plt
from methodes.utils import load_grey, extend_symmetric
from methodes.solver import compute_f_hat, gradient_hat, solve_poisson
from methodes.collage import colle_gradient, colle_pixels, disque

# même moyenne et même écart-type que f (comme dans l'article)
def recale(u, f):
    return (u - u.mean()) / u.std() * f.std() + f.mean()

def mesures(u, f):
    mse = np.mean((u - f)**2)
    psnr = 10 * np.log10(255**2 / mse)
    md = np.abs(u - f).max()
    return mse, psnr, md


# --- 1) éq. 18 : Δu = Δf avec Neumann, on doit retrouver f (table 1)
print(f"{'image':<14}{'gradient':<14}{'MSE':>12}{'PSNR':>10}{'MD':>12}")
for nom in ["prairie.jpg", "concert.jpg", "chat.jpg", "jardin.jpg", "camera.png", "irm.tif"]:
    f = load_grey(os.path.join(RACINE, "images", nom))
    H, W = f.shape

    # gradient exact (éq. 13), comme dans l'article
    V1_hat, V2_hat = gradient_hat(compute_f_hat(f))
    u = recale(solve_poisson(V1_hat, V2_hat, 0, H, W), f)
    mse, psnr, md = mesures(u, f)
    print(f"{nom:<14}{'spectral':<14}{mse:12.2e}{psnr:10.1f}{md:12.2e}")

    # pour comparer : gradient aux différences finies (sur le domaine étendu) donné au même solveur
    fe = extend_symmetric(f)
    D1 = np.roll(fe, -1, axis=0) - fe
    D2 = np.roll(fe, -1, axis=1) - fe
    u = solve_poisson(np.fft.fft2(D1), np.fft.fft2(D2), 0, H, W)
    mse, psnr, md = mesures(recale(u, f), f)
    print(f"{'':<14}{'diff. finies':<14}{mse:12.2e}{psnr:10.1f}{md:12.2e}")


# --- 2) figure 9 : texture haute fréquence collée sur une texture basse fréquence
H, W = 256, 256
j = np.arange(H).reshape(H, 1)
l = np.arange(W).reshape(1, W)
f = 128 + 40 * np.sin(2 * np.pi * l / 90) * np.cos(2 * np.pi * j / 70)          # basse fréquence
g = 128 + 40 * np.cos(2 * np.pi * (j + l) / 6) + 30 * np.cos(2 * np.pi * (j - 2 * l) / 5)   # haute fréquence
cj, cl, r = 128, 128, 60
m = disque(H, W, cj, cl, r)
u = colle_gradient(f, g, m)

# écart |u - f| hors du masque, en fonction de la distance d au bord du disque
d = np.sqrt((j - cj)**2 + (l - cl)**2) - r
e = np.abs(u - f)
ds = np.arange(0, 60)
e_max = [e[(d >= k) & (d < k + 1)].max() for k in ds]
print("écart max hors du masque : d < 5 px :", max(e_max[:5]), " d > 20 px :", max(e_max[20:]))

fig, ax = plt.subplots(2, 3, figsize=(15, 9))
for a, im, t in zip(ax[0], [f, g, u], ["f : basse fréquence", "g : haute fréquence", "collage du gradient (éq. 16)"]):
    a.imshow(im, cmap="gray", vmin=0, vmax=255)
    a.set_title(t)
    a.axis("off")
ax[1, 0].imshow(np.where(m, np.nan, e), cmap="hot")
ax[1, 0].set_title("|u - f| hors du masque")
ax[1, 0].axis("off")
ax[1, 1].semilogy(ds, e_max)
ax[1, 1].set_xlabel("distance au bord du masque (pixels)")
ax[1, 1].set_title("écart max |u - f| à cette distance")
ax[1, 2].plot(f[cj], "k", label="f")
ax[1, 2].plot(colle_pixels(f, g, m)[cj], color="0.7", label="collage des pixels")
ax[1, 2].plot(u[cj], "b", label="collage du gradient")
ax[1, 2].axvline(cl - r, ls=":", color="k")
ax[1, 2].axvline(cl + r, ls=":", color="k")
ax[1, 2].set_title(f"ligne {cj}")
ax[1, 2].legend()
plt.tight_layout()
plt.show()
