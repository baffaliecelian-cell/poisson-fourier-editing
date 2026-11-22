import numpy as np
from PIL import Image
from methodes.utils import extend_symmetric
from methodes.solver import compute_f_hat, gradient_hat, solve_poisson
from methodes.solver3d import extend_symmetric_3d, compute_f_hat_3d, gradient_hat_3d, solve_poisson_3d


# charge l'image en couleur (sert à recolorer le résultat à la fin)
def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"), dtype=float)


# rehaussement des zones sombres (section 4.1, éq. 15)
def contrast_enhance(f, T=50, alpha=2.5):
    H, W = f.shape
    f_hat = compute_f_hat(f)                 # f prolongée en 2H x 2W, en Fourier (éq. 9)
    V1_hat, V2_hat = gradient_hat(f_hat)     # gradient de f (éq. 13)
    V1 = np.fft.ifft2(V1_hat).real           # retour en espace pour le modifier pixel par pixel
    V2 = np.fft.ifft2(V2_hat).real
    X = extend_symmetric(f) < T              # zone sombre, sur le domaine prolongé
    V1[X] = alpha * V1[X]                    # éq. 15 : alpha * grad f dans X, grad f ailleurs
    V2[X] = alpha * V2[X]
    return solve_poisson(np.fft.fft2(V1), np.fft.fft2(V2), f_hat[0, 0], H, W)


# même chose sur un volume H x W x D
def contrast_enhance_3d(f, T=50, alpha=2.5):
    H, W, D = f.shape
    f_hat = compute_f_hat_3d(f)
    V = [np.fft.ifftn(c).real for c in gradient_hat_3d(f_hat)]   # les 3 composantes du gradient
    X = extend_symmetric_3d(f) < T
    for c in V:
        c[X] = alpha * c[X]
    V_hat = [np.fft.fftn(c) for c in V]
    return solve_poisson_3d(*V_hat, f_hat[0, 0, 0], H, W, D)


# étirement final : u a la même moyenne que f, donc le sombre passe sous 0 et le clair au-dessus
# de 255. On décale u pour que son minimum soit 0, puis on le multiplie pour que son maximum soit 255.
# L'article ne parle pas de cette étape, mais ses figures 3 à 5 sont plus claires que u.
def etire(u):
    return (u - u.min()) * 255 / (u.max() - u.min())


# recolore : R, G et B sont multipliés par u / f (fin de la section 4.1)
def color_from_grey(img, f, u):
    ratio = u / np.maximum(f, 1)             # le max évite de diviser par 0
    out = img * ratio[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


# seuil automatique : la zone {f <= T} couvre 25 % de l'image (section 4.1)
def auto_threshold(f, area=0.25):
    return np.percentile(f, 100 * area)
