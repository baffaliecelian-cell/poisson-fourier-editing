import numpy as np
from PIL import Image
#charge l'image et la convertit en gris
def load_grey(path):
    img=Image.open(path)
    img=img.convert("L")
    arr = np.array(img, dtype=float)
    return arr

#prolonge l'image par symétrie miroir -> taille 2H x 2W (condition de Neumann, §3.1)
def extend_symmetric(f):
    H, W = f.shape
    return np.pad(f, ((0, H), (0, W)), mode="symmetric")

