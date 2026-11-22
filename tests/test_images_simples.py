import os, sys
RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from methodes.utils import grad
from methodes.solver import compute_f_hat, gradient_hat, solve_poisson
from methodes.contraste import contrast_enhance

N = 128
rng = np.random.default_rng(0)

# carré clair (200) sur fond sombre (50)
def carre(N=128, a=40, b=88):
    f = np.full((N, N), 50.0)
    f[a:b, a:b] = 200
    return f

# gradient spectral (éq. 13) ramené sur l'image H x W
def grad_spectral(f):
    H, W = f.shape
    V1_hat, V2_hat = gradient_hat(compute_f_hat(f))
    V1 = np.fft.ifft2(V1_hat).real[:H, :W]
    V2 = np.fft.ifft2(V2_hat).real[:H, :W]
    return V1, V2

# V = grad f -> doit redonner f
def reconstruit(f):
    H, W = f.shape
    f_hat = compute_f_hat(f)
    return solve_poisson(*gradient_hat(f_hat), f_hat[0, 0], H, W)

# valeur max du gradient loin des bords du carré (là où il devrait être nul)
def oscillations(G, f, d=4):
    loin = np.ones_like(f, dtype=bool)
    Gx, Gy = grad(f)
    bord = (np.abs(Gx) + np.abs(Gy)) > 1
    for dj in range(-d, d + 1):
        for dl in range(-d, d + 1):
            loin &= ~np.roll(bord, (dj, dl), axis=(0, 1))
    return np.abs(G[loin]).max()


# test 1 : bord net / bord flou -> le gradient spectral oscille autour d'un bord net (Gibbs)
def test_net_flou():
    print("=== test 1 : bord net / bord flou ===")
    images = [("net", carre()), ("flou sigma=1", gaussian_filter(carre(), 1)), ("flou sigma=3", gaussian_filter(carre(), 3))]
    fig, ax = plt.subplots(3, 3, figsize=(13, 11))
    for i, (nom, f) in enumerate(images):
        V1, V2 = grad_spectral(f)
        gx, gy = grad(f)
        err = np.abs(reconstruit(f) - f).max()
        print(f"{nom:13s} oscillations loin du bord : spectral = {oscillations(V2, f):6.2f}"
              f" | diff. finies = {oscillations(gx, f):4.2f} | reconstruction |u-f| = {err:.1e}")
        ax[i, 0].imshow(f, cmap="gray"); ax[i, 0].set_title(nom)
        ax[i, 1].imshow(np.hypot(V1, V2), cmap="gray", vmax=40); ax[i, 1].set_title("|grad| spectral")
        ax[i, 2].plot(V2[64, 20:70], "o-", label="spectral")
        ax[i, 2].plot(gx[64, 20:70], "s--", label="diff. finies")
        ax[i, 2].set_title("dérivée horizontale, ligne 64"); ax[i, 2].legend()
        ax[i, 0].axis("off"); ax[i, 1].axis("off")
    plt.tight_layout()
    plt.show()


# test 2 : bruit -> le solveur reste exact, mais la dérivée et alpha amplifient le bruit
def test_bruit():
    print("=== test 2 : bruit ===")
    sigma = 4
    f = carre() + rng.normal(0, sigma, (N, N))
    V1, V2 = grad_spectral(f)
    gx, gy = grad(f)
    plat = np.zeros((N, N), dtype=bool); plat[5:30, 5:30] = True      # zone sans bord
    print(f"bruit sur l'image : sigma = {f[plat].std():.2f}")
    print(f"bruit sur la dérivée : spectral = {V2[plat].std():.2f} | diff. finies = {gx[plat].std():.2f}")
    print(f"reconstruction V = grad f : |u - f| = {np.abs(reconstruit(f) - f).max():.1e}")

    # contraste : le fond (50) est sous le seuil, son bruit est multiplié par alpha
    u = contrast_enhance(f, T=100, alpha=2.5)
    print(f"bruit dans la zone sombre après contraste : {f[plat].std():.2f} -> {u[plat].std():.2f}"
          f" (x{u[plat].std() / f[plat].std():.2f})")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(f, cmap="gray", vmin=0, vmax=255); ax[0].set_title(f"carré + bruit (sigma = {sigma})")
    ax[1].imshow(np.hypot(V1, V2), cmap="gray", vmax=40); ax[1].set_title("|grad| spectral")
    ax[2].imshow(u, cmap="gray", vmin=0, vmax=255); ax[2].set_title("contraste, T = 100, alpha = 2.5")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.show()


# test 3 : moitié sombre rayée / moitié claire -> pourquoi le contraste assombrit le sombre
def test_bord_contraste():
    print("=== test 3 : contraste sur moitié sombre / moitié claire ===")
    l = np.arange(N)
    rayures = 5 * np.sin(2 * np.pi * l / 8) * (l < N // 2 - 8)
    marche_nette = np.where(l < N // 2, 0, 120.0)
    marche_douce = 60 * (1 + np.tanh((l - N // 2) / 4))           # même marche, étalée sur ~10 pixels
    images = [("bord net", np.tile(40 + rayures + marche_nette, (N, 1))),
              ("bord doux", np.tile(40 + rayures + marche_douce, (N, 1)))]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for a, (nom, f) in zip(ax, images):
        u = contrast_enhance(f, T=100, alpha=2.5)
        sombre = slice(5, N // 2 - 12)
        print(f"{nom:18s} zone sombre : moyenne {f[64, sombre].mean():.1f} -> {u[64, sombre].mean():.1f}"
              f" | amplitude des rayures {np.ptp(f[64, sombre]):.1f} -> {np.ptp(u[64, sombre]):.1f}")
        a.plot(f[64], label="f (avant)")
        a.plot(u[64], label="u (après)")
        a.axhline(0, color="k", lw=0.5); a.axvline(N // 2, color="gray", ls=":")
        a.set_title(f"{nom} : profil de la ligne 64"); a.legend()
    plt.tight_layout()
    plt.show()


test_net_flou()
test_bruit()
test_bord_contraste()
