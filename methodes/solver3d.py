import numpy as np

# même méthode que solver.py, pour un volume H x W x D


# prolonge le volume par miroir sur les 3 axes : 2H x 2W x 2D (condition de Neumann)
def extend_symmetric_3d(f):
    H, W, D = f.shape
    return np.pad(f, ((0, H), (0, W), (0, D)), mode="symmetric")


# pulsations 2 pi m / J, 2 pi n / L, 2 pi p / P, avec les indices recentrés comme en 2D
def pulsations_3d(J, L, P):
    m = np.arange(J).reshape(J, 1, 1)
    n = np.arange(L).reshape(1, L, 1)
    p = np.arange(P).reshape(1, 1, P)
    m = np.where(m < J / 2, m, m - J)
    n = np.where(n < L / 2, n, n - L)
    p = np.where(p < P / 2, p, p - P)
    return 2 * np.pi * m / J, 2 * np.pi * n / L, 2 * np.pi * p / P


# coefficients de Fourier de f sur le domaine prolongé
def compute_f_hat_3d(f):
    return np.fft.fftn(extend_symmetric_3d(f))


# gradient en Fourier : une composante par axe
def gradient_hat_3d(f_hat):
    wm, wn, wp = pulsations_3d(*f_hat.shape)
    return 1j * wm * f_hat, 1j * wn * f_hat, 1j * wp * f_hat


# résout Delta u = div V en Fourier, revient en espace et garde le huitième H x W x D
def solve_poisson_3d(V1_hat, V2_hat, V3_hat, f_hat000, H, W, D):
    wm, wn, wp = pulsations_3d(*V1_hat.shape)
    denom = wm**2 + wn**2 + wp**2
    denom[0, 0, 0] = 1                       # évite 0 / 0
    u_hat = -1j * (wm * V1_hat + wn * V2_hat + wp * V3_hat) / denom
    u_hat[0, 0, 0] = f_hat000                # même moyenne que f
    u = np.fft.ifftn(u_hat)
    return u.real[:H, :W, :D]
