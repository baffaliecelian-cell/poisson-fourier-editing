import numpy as np
from utils import extend_symmetric
#coefficients de Fourier de f sur le domaine quadruplé 
def compute_f_hat(f):
    fe = extend_symmetric(f)   # image 2H x 2W, symétrique et périodique
    f_hat = np.fft.fft2(fe)    # éq. (9)
    return f_hat

#coefficients de Fourier du gradient de f (éq. 13)
def gradient_hat(f_hat):
    J, L = f_hat.shape
    m = np.arange(J).reshape(J, 1)       # m = 0..J-1, en colonne (lignes de f_hat)
    n = np.arange(L).reshape(1, L)       # n = 0..L-1, en ligne (colonnes de f_hat)
    m = np.where(m < J/2, m, m - J)      # recentrage : m -> m-J (band limited)
    n = np.where(n < L/2, n, n - L)
    V1_hat = 2j * np.pi * m / J * f_hat  # éq. (13) : V̂1_mn = (2πim/J) f̂_mn
    V2_hat = 2j * np.pi * n / L * f_hat  # éq. (13) : V̂2_mn = (2πin/L) f̂_mn
    return V1_hat, V2_hat

#résout Δu = div V en Fourier (éq. 12), revient en espace image (éq. 10) et recadre
def solve_poisson(V1_hat, V2_hat, f_hat00, H, W):
    J, L = V1_hat.shape
    m = np.arange(J).reshape(J, 1)
    n = np.arange(L).reshape(1, L)
    m = np.where(m < J/2, m, m - J)          # même recentrage qu'à l'éq. 13
    n = np.where(n < L/2, n, n - L)
    wm = 2 * np.pi * m / J
    wn = 2 * np.pi * n / L
    denom = wm**2 + wn**2                    # dénominateur de l'éq. (12), tableau J x L
    denom[0, 0] = 1                          # évite 0/0 en (m,n) = (0,0)
    u_hat = -1j * (wm * V1_hat + wn * V2_hat) / denom   # éq. (12)
    u_hat[0, 0] = f_hat00                    # constante libre : même moyenne que f
    u = np.fft.ifft2(u_hat)                  # éq. (10)
    return u.real[:H, :W]                    # restriction au domaine initial