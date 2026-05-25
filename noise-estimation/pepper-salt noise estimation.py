# -*- coding: utf8 -*-
import numpy as np

from noise_utils import add_noise, read_grayscale_image, wavelet_hh


image = read_grayscale_image("../image/egypt.jpg", (1080, 1080))
origin_cD = wavelet_hh(image)

amp_range = 250
bins = 10
h_orig = np.histogram(origin_cD, bins=bins, range=(0, amp_range), density=True)[0]

corr_coeffs = []
RNG = np.random.default_rng(0)

for i in range(5, 45, 5):
    pep_salt_image = add_noise(image, "pepper_salt", i, RNG)
    cD = wavelet_hh(pep_salt_image)
    h_noise = np.histogram(cD, bins=bins, range=(0, amp_range), density=True)[0]
    corr_coeffs.append(np.corrcoef(h_orig, h_noise)[0, 1])

print(corr_coeffs)
