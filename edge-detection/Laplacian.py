import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as signal  #

#
suanzi1 = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]])

#
suanzi2 = np.array([[1, 1, 1],
                    [1, -8, 1],
                    [1, 1, 1]])

#
image = Image.open("../image/church.jpg").convert("L")
image_array = np.array(image)

#
image_suanzi1 = signal.convolve2d(image_array, suanzi1, mode="same")
image_suanzi2 = signal.convolve2d(image_array, suanzi2, mode="same")

#
image_suanzi1 = (image_suanzi1 / float(image_suanzi1.max())) * 255
image_suanzi2 = (image_suanzi2 / float(image_suanzi2.max())) * 255

#
image_suanzi1[image_suanzi1 > image_suanzi1.mean()] = 255
image_suanzi2[image_suanzi2 > image_suanzi2.mean()] = 255

#
plt.subplot(2, 1, 1)
plt.imshow(image_array, cmap=cm.gray)
plt.axis("off")
plt.subplot(2, 2, 3)
plt.imshow(image_suanzi1, cmap=cm.gray)
plt.axis("off")
plt.subplot(2, 2, 4)
plt.imshow(image_suanzi2, cmap=cm.gray)
plt.axis("off")
plt.show()