# -*- coding: utf8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pywt

from noise_utils import add_noise, read_grayscale_image


image = read_grayscale_image("../image/church.jpg", (1080, 1080))
RNG = np.random.default_rng(0)

pep_salt_image = add_noise(image, "pepper_salt", 25, RNG)
cv2.imwrite("./pepper-salt noise image.jpg", pep_salt_image)
_, (_, _, cD) = pywt.dwt2(pep_salt_image, "haar")
HH_salt_image = cD.astype(np.int64)

amp_range = 400
Pcoeff_salt_image = np.zeros((2 * amp_range, 1))
for row in HH_salt_image:
    for coefficient in row:
        index = coefficient + amp_range - 1
        if 0 <= index < 2 * amp_range:
            Pcoeff_salt_image[index, 0] += 1

gauss_image = add_noise(image, "gauss", 25, RNG)
cv2.imwrite("./gauss noise image.jpg", gauss_image)
_, (_, _, cD) = pywt.dwt2(gauss_image, "haar")
HH_gauss_image = cD.astype(np.int64)

Pcoeff_gauss_image = np.zeros((2 * amp_range, 1))
for row in HH_gauss_image:
    for coefficient in row:
        index = coefficient + amp_range - 1
        if 0 <= index < 2 * amp_range:
            Pcoeff_gauss_image[index, 0] += 1

x = np.arange(-amp_range, amp_range, 1)
plt.plot(x, Pcoeff_salt_image, color="k")
plt.xlabel("Coe")
plt.ylabel("Pcoe")
plt.savefig("HH coefficient distribution by pepper-salt noise.jpg")
plt.clf()

plt.plot(x, Pcoeff_gauss_image, color="k")
plt.xlabel("Coe")
plt.ylabel("Pcoe")
plt.savefig("HH coefficient distribution by gauss noise.jpg")
plt.clf()
