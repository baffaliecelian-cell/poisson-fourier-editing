import numpy as np
from PIL import Image


# charge une image en niveaux de gris
def load_grey(path):
    return np.array(Image.open(path).convert("L"), dtype=float)


# prolonge l'image par miroir : taille 2H x 2W (condition de Neumann, section 3.1)
def extend_symmetric(f):
    H, W = f.shape
    return np.pad(f, ((0, H), (0, W)), mode="symmetric")


# gradient par différences finies (pixel suivant moins pixel courant), 0 sur le dernier bord
def grad(u):
    gx = np.zeros_like(u)
    gy = np.zeros_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    return gx, gy
