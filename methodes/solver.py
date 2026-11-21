import numpy as np
from methodes.utils import extend_symmetric


# coefficients de Fourier de f sur le domaine prolongé 2H x 2W (éq. 9)
def compute_f_hat(f):
    fe = extend_symmetric(f)     # image miroir : périodique et sans saut aux raccords
    return np.fft.fft2(fe)


# indices de fréquence ramenés entre -J/2 et J/2 : la FFT range m = 0..J-1,
# mais les m > J/2 sont en fait les fréquences négatives m - J
def recentre(J, L):
    m = np.arange(J).reshape(J, 1)
    n = np.arange(L).reshape(1, L)
    m = np.where(m < J / 2, m, m - J)
    n = np.where(n < L / 2, n, n - L)
    return m, n


# gradient de f en Fourier (éq. 13) : dériver revient à multiplier par i w
def gradient_hat(f_hat):
    J, L = f_hat.shape
    m, n = recentre(J, L)
    V1_hat = 2j * np.pi * m / J * f_hat      # dérivée selon les lignes
    V2_hat = 2j * np.pi * n / L * f_hat      # dérivée selon les colonnes
    return V1_hat, V2_hat


# résout Delta u = div V en Fourier (éq. 12), revient en espace (éq. 10) et garde le quart H x W
def solve_poisson(V1_hat, V2_hat, f_hat00, H, W):
    J, L = V1_hat.shape
    m, n = recentre(J, L)
    wm = 2 * np.pi * m / J
    wn = 2 * np.pi * n / L
    denom = wm**2 + wn**2
    denom[0, 0] = 1                          # en (0, 0) l'éq. 12 donne 0 / 0
    u_hat = -1j * (wm * V1_hat + wn * V2_hat) / denom
    u_hat[0, 0] = f_hat00                    # la constante n'est pas fixée par le gradient : on garde la moyenne de f
    u = np.fft.ifft2(u_hat)
    return u.real[:H, :W]
