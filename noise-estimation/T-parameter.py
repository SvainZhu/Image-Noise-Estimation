# -*- coding: utf8 -*-
import matplotlib.pyplot as plt
import numpy as np

from noise_utils import add_noise, insignificant_energy_ratio, read_grayscale_image, wavelet_hh


image = read_grayscale_image("../image/church.jpg", (1080, 1080))
RNG = np.random.default_rng(0)

pep_salt_image_coeffs = []
gauss_image_coeffs = []

for i in range(0, 24, 4):
    pep_salt_image = add_noise(image, "pepper_salt", i, RNG)
    pep_salt_image_coeffs.append(wavelet_hh(pep_salt_image))

    gauss_image = add_noise(image, "gauss", i, RNG)
    gauss_image_coeffs.append(wavelet_hh(gauss_image))

pep_salt_ER = np.zeros((6, 9))
gauss_ER = np.zeros((6, 9))

for i in range(6):
    for j in range(10, 100, 10):
        column = j // 10 - 1
        pep_salt_ER[i, column] = insignificant_energy_ratio(pep_salt_image_coeffs[i], j)
        gauss_ER[i, column] = insignificant_energy_ratio(gauss_image_coeffs[i], j)

x = np.arange(10, 100, 10)
for i in range(6):
    plt.plot(x, pep_salt_ER[i], color="r", linewidth=1)
    plt.plot(x, gauss_ER[i], color="k", linewidth=1)

plt.xlabel("T")
plt.ylabel("ER")
plt.savefig("ER graph by different T parameter.jpg")
