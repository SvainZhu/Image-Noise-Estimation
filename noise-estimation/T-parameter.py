# -*- coding:utf8 -*-
import cv2
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import pywt
#本程序依赖matplotlib, numpy, opencv-python等python库


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


#定义函数去计算图像HH子带系数的能量比
def insignificant_energy_ratio(HH_D, T):     #HH_D为HH子带系数矩阵，T为阈值
    D_T = zeros(HH_D.shape)
    N_T = 0
    for i in range(0, HH_D.shape[0]):
        for j in range(0, HH_D.shape[1]):
            if abs(HH_D[i, j]) < T:
                D_T[i, j] = HH_D[i, j]
                N_T += 1
    ER = (np.sum(D_T**2)/N_T) / (np.sum(HH_D**2)/(HH_D.shape[0]*HH_D.shape[1]))
    return ER

image = cv2.imread("../image/church.jpg")
image = cv2.resize(image, (1080, 1080))
# 将多通道图像变为单通道图像
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image.astype(np.float32)

#添加噪音之后的图像列表
pep_salt_images = []
gauss_images = []

#噪音图像的HH子带系数
pep_salt_image_coeffs = []
gauss_image_coeffs = []
#迭代得到噪音图像及它们的HH子带系数
for i in range(0, 24, 4):
    pep_salt_image = add_noise(image, "pepper_salt", i)
    pep_salt_images.append(pep_salt_image)
    coeffs = pywt.dwt2(pep_salt_image, 'sym4')
    cA, (cH, cV, cD) = coeffs  # 其中cA为图像的LL系数，cH为LH系数，cV为HL系数以及cD为HH系数
    pep_salt_image_coeffs.append(cD)
    gauss_image = add_noise(image, "gauss", i)
    gauss_images.append(gauss_image)
    coeffs = pywt.dwt2(gauss_image, 'sym4')
    cA, (cH, cV, cD) = coeffs  # 其中cA为图像的LL系数，cH为LH系数，cV为HL系数以及cD为HH系数
    gauss_image_coeffs.append(cD)

#噪音图像的HH子带系数的不同阈值下的能量比矩阵
pep_salt_ER = zeros((6, 9))
gauss_ER = zeros((6, 9))
#计算噪音图像的HH子带系数的不同阈值下的能量比矩阵
for i in range(0, 6):
    for j in range(10, 100, 10):
        pep_salt_ER[i, j/10 - 1] = insignificant_energy_ratio(pep_salt_image_coeffs[i], j)
        gauss_ER[i, j/10 - 1] = insignificant_energy_ratio(gauss_image_coeffs[i], j)

#得出不同阈值下椒盐噪音图像的HH子带系数的能量比图像并用红线表示
x = np.arange(10, 100, 10)
plt.plot(x, pep_salt_ER[0], color="r", linewidth=1)
plt.plot(x, pep_salt_ER[1], color="r", linewidth=1)
plt.plot(x, pep_salt_ER[2], color="r", linewidth=1)
plt.plot(x, pep_salt_ER[4], color="r", linewidth=1)
plt.plot(x, pep_salt_ER[5], color="r", linewidth=1)

#得出不同阈值下高斯白噪音图像的HH子带系数的能量比图像并用黑线表示
plt.plot(x, gauss_ER[0], color="k", linewidth=1)
plt.plot(x, gauss_ER[1], color="k", linewidth=1)
plt.plot(x, gauss_ER[2], color="k", linewidth=1)
plt.plot(x, gauss_ER[3], color="k", linewidth=1)
plt.plot(x, gauss_ER[4], color="k", linewidth=1)
plt.plot(x, gauss_ER[5], color="k", linewidth=1)
plt.xlabel("T")
plt.ylabel("ER")
#不同噪音强度及阈值下的HH系数能量比
plt.title("Energy ratio of HH coefficient at different noise levels and thresholds")
plt.savefig("ER graph by different T parameter.jpg")
