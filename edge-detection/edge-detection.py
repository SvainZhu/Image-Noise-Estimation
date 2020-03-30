import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as signal


#
def func(x, y, sigma=1):
    return 100 * (1 / (2 * np.pi * sigma)) * np.exp(-((x - 2) ** 2 + (y - 2) ** 2) / (2.0 * sigma ** 2))


#
suanzi1 = np.fromfunction(func, (5, 5), sigma=5)

#
suanzi2 = np.array([[1, 1, 1],
                    [1, -8, 1],
                    [1, 1, 1]])

#
image = Image.open("../image/church.jpg").convert("L")
image_array = np.array(image)

#
image_blur = signal.convolve2d(image_array, suanzi1, mode="same")

#
image2 = signal.convolve2d(image_blur, suanzi2, mode="same")

#
image2 = (image2 / float(image2.max())) * 255

#
image2[image2 > image2.mean()] = 255

#
plt.subplot(2, 1, 1)
plt.imshow(image_array, cmap=cm.gray)
plt.axis("off")
plt.subplot(2, 1, 2)
plt.imshow(image2, cmap=cm.gray)
plt.axis("off")
plt.show()