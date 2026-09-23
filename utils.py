import numpy as np
from PIL import Image
#charge l'image et la convertit en gris
def load_grey(path):
    img=Image.open(path)
    img=img.convert("L")
    arr = np.array(img, dtype=float)
    return arr
#Calcule le gradient de l'image, renvoie 2 tableaux de  la même taille, avec des 0 en un bord
def grad(u):
    gx=np.zeros_like(u)
    gy=np.zeros_like(u)
    gx[:,:-1]=u[:,1:]-u[:,:-1]
    gy[:-1,:]=u[1:,:]-u[:-1,:]
    return gx, gy

def div(V1, V2):
    dvx = np.zeros_like(V1)
    dvy = np.zeros_like(V2)
    dvx[:, 1:] = V1[:, 1:] - V1[:, :-1]
    dvx[:, 0]  = V1[:, 0]
    dvy[1:, :] = V2[1:, :] - V2[:-1, :]
    dvy[0, :]  = V2[0, :]
    return dvx + dvy

