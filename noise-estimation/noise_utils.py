# -*- coding: utf-8 -*-
"""Shared helpers for image noise experiments."""

from __future__ import annotations

import cv2
import numpy as np
import pywt
import scipy.signal as signal


def read_grayscale_image(path, size=None):
    """Read an image as float32 grayscale data."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError("Image not found: {0}".format(path))
    if size is not None:
        image = cv2.resize(image, size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float32)


def add_noise(image, method, intensity, rng=None):
    """Add pepper-salt or Gaussian noise to an image."""
    rng = rng or np.random.default_rng()
    image = image.astype(np.float32, copy=False)

    if method == "pepper_salt":
        threshold = intensity / 200.0
        salt_noise = np.where(rng.random(image.shape) < threshold, 255, 0)
        pepper_noise = np.where(rng.random(image.shape) < threshold, -255, 0)
        noisy = image + salt_noise.astype(np.float32) + pepper_noise.astype(np.float32)
    elif method == "gauss":
        noisy = image + rng.normal(0, intensity, image.shape).astype(np.float32)
    else:
        raise ValueError("Unsupported noise type: {0}".format(method))

    return np.clip(noisy, 0, 255).astype(np.float32)


def wavelet_hh(image, wavelet="sym4"):
    """Return the HH sub-band coefficients of a single-level 2D DWT."""
    _, (_, _, cD) = pywt.dwt2(image, wavelet)
    return cD


def insignificant_energy_ratio(hh_coefficients, threshold):
    """Calculate the energy ratio of HH coefficients below ``threshold``."""
    hh_coefficients = np.asarray(hh_coefficients, dtype=np.float32)
    mask = np.abs(hh_coefficients) < threshold
    if not np.any(mask):
        return 0.0

    selected_energy = np.mean(np.square(hh_coefficients[mask]))
    total_energy = np.mean(np.square(hh_coefficients))
    if total_energy == 0:
        return 0.0
    return selected_energy / total_energy


def noise_type(energy_ratio):
    """Classify image noise from the HH energy ratio."""
    if 0.5 < energy_ratio <= 1.1:
        return "gauss"
    if 0 <= energy_ratio <= 0.5:
        return "pepper_salt"
    raise ValueError("Energy ratio is outside the supported range: {0}".format(energy_ratio))


def estimate_noise_intensity(noise_hh, origin_hh, method):
    """Estimate noise intensity from HH coefficients."""
    if method == "gauss":
        return float(np.median(np.abs(noise_hh)) / 0.6745)

    if method == "pepper_salt":
        amp_range = 250
        bins = 10
        h_orig = np.histogram(origin_hh, bins=bins, range=(0, amp_range), density=True)[0]
        h_noise = np.histogram(noise_hh, bins=bins, range=(0, amp_range), density=True)[0]
        p = np.corrcoef(h_orig, h_noise)[0, 1]
        if np.isnan(p):
            return 0.0
        return float(524.8 - 1637 * p + 1859 * p**2 - 743.3 * p**3)

    raise ValueError("Unsupported noise type: {0}".format(method))


def gaussian_operator(x, y, sigma=1):
    """Generate values for the Gaussian smoothing operator."""
    return 100 * (1 / (2 * np.pi * sigma)) * np.exp(
        -((x - 2) ** 2 + (y - 2) ** 2) / (2.0 * sigma**2)
    )


def edge_detection(image):
    """Detect image edges using Gaussian smoothing followed by a Laplacian operator."""
    operator_5 = np.fromfunction(gaussian_operator, (5, 5), sigma=5)
    operator_3 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

    image_blur = signal.convolve2d(image, operator_5, mode="same")
    edge_image = signal.convolve2d(image_blur, operator_3, mode="same")
    max_value = edge_image.max()
    if max_value != 0:
        edge_image = (edge_image / float(max_value)) * 255
    edge_image[edge_image > edge_image.mean()] = 255
    return edge_image


def edge_process(image, edge_image):
    """Smooth non-edge pixels with a 3x3 mean filter."""
    smoothed = cv2.blur(image, (3, 3))
    result = image.copy()
    result[edge_image != 255] = smoothed[edge_image != 255]
    return result.astype(np.float32)
