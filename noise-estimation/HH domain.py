# -*- coding: utf8 -*-
import numpy as np
import prettytable as pt

from noise_utils import (
    add_noise,
    edge_detection,
    edge_process,
    estimate_noise_intensity,
    insignificant_energy_ratio,
    noise_type,
    read_grayscale_image,
    wavelet_hh,
)


T = 60
N = 100
IMAGE_SIZE = (192, 192)
RNG = np.random.default_rng(0)

images = []
new_images = []
origin_cD = []

for i in range(N):
    image = read_grayscale_image("../test/" + str(i) + ".jpg", IMAGE_SIZE)
    images.append(image)
    origin_cD.append(wavelet_hh(image))

    image_edge = edge_detection(image)
    new_images.append(edge_process(image, image_edge))


pep_salt_image_intensity = np.zeros((10, N))
gauss_image_intensity = np.zeros((10, N))
pep_salt_intensity_mean = np.zeros((10, 1))
pep_salt_intensity_var = np.zeros((10, 1))
gauss_intensity_mean = np.zeros((10, 1))
gauss_intensity_var = np.zeros((10, 1))

for i in range(2, 22, 2):
    row = i // 2 - 1
    for j in range(N):
        pep_salt_image = add_noise(images[j], "pepper_salt", i, RNG)
        cD = wavelet_hh(pep_salt_image)
        ER = insignificant_energy_ratio(cD, T)
        pep_salt_image_intensity[row, j] = estimate_noise_intensity(
            cD, origin_cD[j], noise_type(ER)
        )

        gauss_image = add_noise(new_images[j], "gauss", i, RNG)
        cD = wavelet_hh(gauss_image)
        ER = insignificant_energy_ratio(cD, T)
        gauss_image_intensity[row, j] = estimate_noise_intensity(
            cD, origin_cD[j], noise_type(ER)
        )

for i in range(10):
    pep_salt_intensity_mean[i, 0] = np.mean(pep_salt_image_intensity[i, :])
    pep_salt_intensity_var[i, 0] = np.var(pep_salt_image_intensity[i, :])
    gauss_intensity_mean[i, 0] = np.mean(gauss_image_intensity[i, :])
    gauss_intensity_var[i, 0] = np.var(gauss_image_intensity[i, :])

tb = pt.PrettyTable()
x = np.arange(2, 22, 2)
tb.add_column(u"  噪声强度实际值  ", x)
tb.add_column(u"  椒盐噪声估计值  ", np.array(pep_salt_intensity_mean).flatten())
tb.add_column(u"椒盐噪声估计方差值 ", np.array(pep_salt_intensity_var).flatten())
tb.add_column(u" 高斯白噪声估计值  ", np.array(gauss_intensity_mean).flatten())
tb.add_column(u"高斯白噪声估计方差值", np.array(gauss_intensity_var).flatten())
print(tb)
