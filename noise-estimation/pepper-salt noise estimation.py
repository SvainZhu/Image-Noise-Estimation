# -*- coding:utf8 -*-
import cv2
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import pywt
from interval import Interval
#本程序依赖matplotlib, numpy, scipy,opencv-python等python库


#给图像添加椒盐噪声及高斯噪声
def add_noise(image, method, intensity):        #image为图像，method为噪音类型，intensit为噪音强度
    if method == "pepper_salt":
        #使用随机函数生成盐噪声
        salt_noise = np.random.randint(0, 256, image.shape)
        salt_noise = np.where(salt_noise < (intensity / 200.0 * 256), 255, 0)
        salt_noise = salt_noise.astype(float32)
        #使用随机函数生成椒噪声
        pepper_noise = np.random.randint(0, 256, image.shape)
        pepper_noise = np.where(pepper_noise < (intensity / 200.0 * 256), -255, 0)
        pepper_noise = pepper_noise.astype(float32)
        #将椒盐噪声添加到图像中去
        image = image + salt_noise + pepper_noise
        image = np.where(image < 0, 0, np.where(image > 255, 255, image))
    elif method == "gauss":
        #使用随机函数生成高斯白噪声，均值为0，方差为噪音强度
        gauss_noise = np.random.normal(0, intensity, image.shape)
        gauss_noise = gauss_noise.astype(float32)
        #将高斯白噪声添加到图像中去
        image = image + gauss_noise
        image = np.where(image < 0, 0, np.where(image > 255, 255, image))
    else:
        raise ("Type Error")
    return image

#读取图像并转换成灰度图
image = cv2.imread("../image/egypt.jpg")
image = cv2.resize(image, (1080, 1080))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image.astype(np.float32)

cA, (cH, cV, cD) = pywt.dwt2(image, 'sym4')         #其中cA为图像的LL系数，cH为LH系数，cV为HL系数以及cD为HH系数
origin_cD = cD
amp_range = 250
H = 10
h_orig = np.histogram(origin_cD, bins=H, range=(0, amp_range), density=True, weights=None)[0]

corr_coeffs = []
#生成不同椒盐强度的噪音图像并进行小波变换
for i in range(5, 45, 5):
    pep_salt_image = add_noise(image, "pepper_salt", i)
    coeffs = pywt.dwt2(pep_salt_image, 'sym4')
    cA, (cH, cV, cD) = coeffs           #其中cA为图像的LL系数，cH为LH系数，cV为HL系数以及cD为HH系数
    h_noise = np.histogram(cD, bins=H, range=(0, amp_range), density=True, weights=None)[0]
    corr_coeffs.append(corrcoef(h_orig, h_noise)[0, 1])


print(corr_coeffs)