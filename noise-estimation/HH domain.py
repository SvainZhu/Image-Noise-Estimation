# -*- coding:utf8 -*-
import cv2
from pylab import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as signal
import pywt
import os
#本程序依赖matplotlib, pillow, numpy, scipy,opencv-python等python库


# 生成高斯算子的函数
def func(x, y, sigma=1):
    return 100 * (1 / (2 * np.pi * sigma)) * np.exp(-((x - 2) ** 2 + (y - 2) ** 2) / (2.0 * sigma ** 2))


#使用改进后的边缘检测算法去检测图像边缘
def edge_detection(image):
    # 生成标准差为5的5*5高斯算子
    operator_5 = np.fromfunction(func, (5, 5), sigma=5)

    # Laplace扩展算子
    operator_3 = np.array([[1, 1, 1],
                           [1, -8, 1],
                           [1, 1, 1]])
    # 利用生成的高斯算子与原图像进行卷积对图像进行平滑处理
    image_blur = signal.convolve2d(image, operator_5, mode="same")

    # 对平滑后的图像进行边缘检测
    edge_image = signal.convolve2d(image_blur, operator_3, mode="same")

    # 结果转化到0-255
    edge_image = (edge_image / float(edge_image.max())) * 255

    # 将大于灰度平均值的灰度值变成255（白色），便于观察边缘
    edge_image[edge_image > edge_image.mean()] = 255
    return edge_image


#定义图像处理函数去平滑图像边缘，使用均值平滑法
def edge_process(image1, image2):    #image1为原始图像，image2为边缘图像
    for i in range(1, image1.shape[0] - 1):
        for j in range(1, image1.shape[1] - 1):
            if image2[i, j] != 255:
                image1[i, j] = (image1[i, j-1] + image1[i, j+1] + image1[i-1, j] + image1[i-1, j-1] + image1[i, j]
                                +image1[i-1, j+1] + image1[i+1, j] + image1[i+1, j+1] + image1[i+1, j-1])/9
    return image1


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


#定义函数去判断图像噪声的类型
def noise_type(ER):
    if ER > 0.5 and ER <= 1.1:
        return "gauss"
    elif ER >= 0 and ER <= 0.5:
        return "pepper_salt"
    else:
        return "Error"

#定义函数去估计图像的噪声
def noise_estimation(noise_cD, origin_cD, method):
    #noise_cD为噪音图像的HH子带系数矩阵,origin_cD为原始图像的HH子带系数矩阵，method为噪音的类型
    if method == "gauss":
        return np.median(abs(noise_cD)) / 0.6745
    elif method == "pepper_salt":
        amp_range = 250
        H = 10
        h_orig = np.histogram(origin_cD, bins=H, range=(0, amp_range), density=True, weights=None)[0]
        h_noise = np.histogram(noise_cD, bins=H, range=(0, amp_range), density=True, weights=None)[0]
        p = corrcoef(h_orig, h_noise)[0, 1]
        return 524.8 - 1637*p +1859*pow(p, 2) - 743.3*pow(p, 3)
        #return -0.222 + 0.004119*p + 1.794e-07*pow(p, 2) + 1.419e-11*pow(p, 3)
    else:
        return "Type Error"

T = 60
images = []
new_images = []
origin_cD = []
N = 100
for i in range(0, N, 1):
    image = cv2.imread("../test/" + str(i) + ".jpg")
    image = cv2.resize(image, (192, 192))
    # 将多通道图像变为单通道图像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)
    images.append(image)
    cA, (cH, cV, cD) = pywt.dwt2(image, 'sym4')         #其中cA为图像的LL系数，cH为LH系数，cV为HL系数以及cD为HH系数
    origin_cD.append(cD)

    image_edge = edge_detection(image)
    new_image = edge_process(image, image_edge)
    new_image = new_image.astype(np.float32)
    new_images.append(new_image)


#噪音图像的HH子带系数
pep_salt_image_intensity = zeros((10, N))
gauss_image_intensity = zeros((10, N))
pep_salt_intensity_mean = zeros((10, 1))
pep_salt_intensity_var = zeros((10, 1))

gauss_intensity_mean = zeros((10, 1))
gauss_intensity_var = zeros((10, 1))
#迭代得到噪音图像及它们的HH子带系数
for i in range(2, 22, 2):
    for j in range(0, N, 1):
        pep_salt_image = add_noise(images[j], "pepper_salt", i)
        coeffs = pywt.dwt2(pep_salt_image, 'sym4')
        cA, (cH, cV, cD) = coeffs           #其中cA为图像的LL系数，cH为LH系数，cV为HL系数以及cD为HH系数
        ER = insignificant_energy_ratio(cD, T)
        pep_salt_image_intensity[i/2-1, j] = noise_estimation(cD, origin_cD[j], noise_type(ER))


        gauss_image = add_noise(new_images[j], "gauss", i)
        coeffs = pywt.dwt2(gauss_image, 'sym4')
        cA, (cH, cV, cD) = coeffs           # 其中cA为图像的LL系数，cH为LH系数，cV为HL系数以及cD为HH系数
        ER = insignificant_energy_ratio(cD, T)
        gauss_image_intensity[i/2-1, j] = noise_estimation(cD, origin_cD[j], noise_type(ER))

for i in range(0, 10, 1):
    pep_salt_intensity_mean[i, 0] = np.mean(pep_salt_image_intensity[i, :])
    pep_salt_intensity_var[i, 0] = np.var(pep_salt_image_intensity[i, :])
    gauss_intensity_mean[i, 0] = np.mean(gauss_image_intensity[i, :])
    gauss_intensity_var[i, 0] = np.var(gauss_intensity_var[i, :])


print(pep_salt_intensity_mean)
print(gauss_intensity_mean)