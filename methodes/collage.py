import numpy as np
from methodes.utils import extend_symmetric
from methodes.solver import compute_f_hat, gradient_hat, solve_poisson

# gradient spectral de f (éq. 13), en espace image, sur le domaine étendu 2H x 2W
def gradient(f):
    V1_hat, V2_hat = gradient_hat(compute_f_hat(f))
    return np.fft.ifft2(V1_hat).real, np.fft.ifft2(V2_hat).real

# éq. 16 : V = grad g dans le masque, grad f ailleurs, puis Poisson
# f, g et masque ont la même taille H x W
def colle_gradient(f, g, masque):
    H, W = f.shape
    V1, V2 = gradient(f)
    G1, G2 = gradient(g)
    M = extend_symmetric(masque)               # le masque doit avoir la taille du domaine étendu
    V1[M] = G1[M]
    V2[M] = G2[M]
    u = solve_poisson(np.fft.fft2(V1), np.fft.fft2(V2), 0, H, W)
    return u + np.mean(f[~masque] - u[~masque])   # on recale u sur f en dehors du masque

# copie naïve des pixels, pour comparer
def colle_pixels(f, g, masque):
    if f.ndim == 3:
        masque = masque[:, :, None]
    return np.where(masque, g, f)

# version couleur : un canal après l'autre
def colle_couleur(img, g, masque):
    canaux = [colle_gradient(img[:, :, c], g[:, :, c], masque) for c in range(3)]
    return np.clip(np.stack(canaux, axis=2), 0, 255).astype(np.uint8)

# source prise dans la même image : g[j, l] = f[j + dy, l + dx]
def decale(f, dy, dx):
    return np.roll(f, (-dy, -dx), axis=(0, 1))

# source prise dans une autre image : on pose la petite image src dans une image H x W,
# coin haut-gauche en (top, left), en prolongeant ses bords autour (pas de saut, donc pas de Gibbs)
def pose(src, H, W, top, left):
    h, w = src.shape[:2]
    bords = ((top, H - top - h), (left, W - left - w)) + ((0, 0),) * (src.ndim - 2)
    return np.pad(src, bords, mode="edge")

# masque en forme de disque
def disque(H, W, cj, cl, r):
    j = np.arange(H).reshape(H, 1)
    l = np.arange(W).reshape(1, W)
    return (j - cj)**2 + (l - cl)**2 < r**2
