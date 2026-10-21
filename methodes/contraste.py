import numpy as np
from PIL import Image
from methodes.utils import extend_symmetric
from methodes.solver import compute_f_hat, gradient_hat, solve_poisson

#charge l'image en couleur (pour recolorer le résultat à la fin)
def load_rgb(path):
    img = Image.open(path)
    img = img.convert("RGB")
    arr = np.array(img, dtype=float)
    return arr

#rehaussement de contraste local (section 4.1, éq. 15)
def contrast_enhance(f, T=50, alpha=2.5):
    H, W = f.shape
    f_hat = compute_f_hat(f)                 # f̂ sur le domaine quadruplé (éq. 9)
    V1_hat, V2_hat = gradient_hat(f_hat)     # ∇f en Fourier (éq. 13)
    V1 = np.fft.ifft2(V1_hat).real           # ∇f en espace image, taille 2H x 2W
    V2 = np.fft.ifft2(V2_hat).real
    X = extend_symmetric(f) < T              # zone sombre, sur le domaine quadruplé
    V1[X] = alpha * V1[X]                    # éq. (15) : V = α∇f dans X
    V2[X] = alpha * V2[X]                    #            V = ∇f ailleurs
    V1_hat = np.fft.fft2(V1)                 # coefficients de Fourier du champ modifié
    V2_hat = np.fft.fft2(V2)
    return solve_poisson(V1_hat, V2_hat, f_hat[0, 0], H, W)

#recolore : chaque canal R, G, B est multiplié par u/f (fin de la section 4.1)
def color_from_grey(img, f, u):
    ratio = u / np.maximum(f, 1)          # facteur d'éclaircissement par pixel, évite /0
    out = img * ratio[:, :, None]         # même facteur sur R, G, B
    out = np.clip(out, 0, 255)            # u peut sortir de [0, 255]
    return out.astype(np.uint8)

#seuil automatique : {f <= T} couvre 25 % de l'image (section 4.1)
def auto_threshold(f, area=0.25):
    return np.percentile(f, 100 * area)
