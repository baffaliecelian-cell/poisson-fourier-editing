import numpy as np
from utils import extend_symmetric
#coefficients de Fourier de f sur le domaine quadruplé (§3.1 + éq. 9)
def compute_f_hat(f):
    fe = extend_symmetric(f)   # image 2H x 2W, symétrique et périodique
    f_hat = np.fft.fft2(fe)    # éq. (9)
    return f_hat