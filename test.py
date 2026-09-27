import numpy as np
import matplotlib.pyplot as plt
from utils import load_grey, grad
from solver import compute_f_hat, gradient_hat, solve_poisson

def test_reconstruction(path):
    f = load_grey(path)
    H, W = f.shape

    # chaîne complète : f -> f_hat -> V_hat -> u_hat -> u
    f_hat = compute_f_hat(f)
    V1_hat, V2_hat = gradient_hat(f_hat)
    u = solve_poisson(V1_hat, V2_hat, f_hat[0, 0], H, W)

    # 1) V = grad f doit redonner f
    err = np.abs(u - f).max()
    print("erreur max |u - f| :", err)

    # 2) le gradient spectral doit être réel (vérifie le recentrage)
    V1 = np.fft.ifft2(V1_hat)
    V2 = np.fft.ifft2(V2_hat)
    print("partie imaginaire max du gradient :", max(np.abs(V1.imag).max(), np.abs(V2.imag).max()))

    # 3) comparaison avec les différences finies (V1 vertical = gy, V2 horizontal = gx)
    gx, gy = grad(f)
    V1 = V1.real[:H, :W]
    V2 = V2.real[:H, :W]

    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    ax[0, 0].imshow(f, cmap="gray");            ax[0, 0].set_title("f")
    ax[0, 1].imshow(u, cmap="gray");            ax[0, 1].set_title("u reconstruite")
    ax[0, 2].imshow(np.abs(u - f), cmap="hot"); ax[0, 2].set_title(f"|u - f|, max = {err:.1e}")
    ax[1, 0].imshow(V2, cmap="gray");           ax[1, 0].set_title("V2 spectral (horizontal)")
    ax[1, 1].imshow(gx, cmap="gray");           ax[1, 1].set_title("gx différences finies")
    ax[1, 2].imshow(V1, cmap="gray");           ax[1, 2].set_title("V1 spectral (vertical)")
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout()
    plt.show()

test_reconstruction("KAZEA36IWVEMRNZ55KCRLLFPBY.jpg")