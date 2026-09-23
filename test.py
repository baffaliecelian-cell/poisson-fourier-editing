from utils import load_grey, grad, div
import matplotlib.pyplot as plt

img = load_grey("KAZEA36IWVEMRNZ55KCRLLFPBY.jpg")
gx, gy = grad(img)

print(img.shape, img.dtype, img.min(), img.max())
print(gx.shape, gy.shape)

# gradient horizontal
plt.imshow(gx, cmap="gray"); plt.title("gradient x"); plt.show()

# test de div : div(grad(img)) doit donner le Laplacien (les contours ressortent)
lap = div(gx, gy)
print("Laplacien:", lap.shape)
plt.imshow(lap, cmap="gray"); plt.title("div(grad) = Laplacien"); plt.show()