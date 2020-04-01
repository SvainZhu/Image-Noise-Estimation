# -*- coding:utf8 -*-
import cv2
from pylab import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as signal
import pywt
#本程序依赖matplotlib, pillow, numpy, scipy,opencv-python等python库


# 生成高斯算子的函数
def func(x, y, sigma=1):
    return 100 * (1 / (2 * np.pi * sigma)) * np.exp(-((x - 2) ** 2 + (y - 2) ** 2) / (2.0 * sigma ** 2))


#使用改进后的边缘检测算法去检测图像边缘
def edge_detection(image):
    # 生成标准差为5的5*5高斯算子
    suanzi1 = np.fromfunction(func, (5, 5), sigma=5)

    # Laplace扩展算子
    suanzi2 = np.array([[1, 1, 1],
                        [1, -8, 1],
                        [1, 1, 1]])

    # 打开图像并转化成灰度图像
    image = Image.open(image).convert("L")
    image_array = np.array(image)

    # 利用生成的高斯算子与原图像进行卷积对图像进行平滑处理
    image_blur = signal.convolve2d(image_array, suanzi1, mode="same")

    # 对平滑后的图像进行边缘检测
    image2 = signal.convolve2d(image_blur, suanzi2, mode="same")

    # 结果转化到0-255
    image2 = (image2 / float(image2.max())) * 255

    # 将大于灰度平均值的灰度值变成255（白色），便于观察边缘
    image2[image2 > image2.mean()] = 255
    return image2


#定义图像处理函数去平滑图像边缘，使用均值平滑法
def edge_process(image1, image2):    #image1为原始图像，image2为边缘图像
    for i in range(1, image1.shape[0] - 2):
        for j in range(1, image1.shape[1] - 2):
            if image2[i,j] != 255:
                image1[i,j] = (image1[i,j-1] + image1[i,j+1] + image1[i-1,j] + image1[i+1,j])/4
    return image1


#给图像添加椒盐噪声及高斯噪声
def add_noise(image, method, intensity):        #image为图像，method为噪音类型，intensit为噪音强度
    if method == "pepper_salt":
        noise_salt = np.random.randint(0, 256, image.shape)
        noise_salt = np.where(noise_salt < (intensity / 200 * 256), 255, 0)
        noise_salt.astype("float")
        noise_pepper = np.random.randint(0, 256, image.shape)
        noise_pepper = np.where(noise_pepper < (intensity / 200 * 256), -255, 0)
        noise_pepper.astype("float")
        image = image + noise_salt + noise_pepper
        image = np.where(image < 0, 0, np.where(image > 255, 255, image))
    elif method == "gauss":
        gauss_noise = np.random.normal(0, intensity, image.shape)
        image = image + gauss_noise
        image = np.where(image < 0, 0, np.where(image > 255, 255, image))
    else:
        raise ("Type Error")
    return image


#定义函数去计算图像HH子带系数的能量比
def insignificant_energy_ratio(HH_D,T):     #HH_D为HH子带系数矩阵，T为阈值
    D_T = zeros(HH_D.shape)
    for i in range (0, HH_D.shape[0] - 1):
        for j in range(0, HH_D.shape[1] - 1):
            if abs(HH_D[i,j]) < T:
                D_T[i,j] = HH_D[i,j]
    ER = (sum(D_T**2)/(D_T.shape[0]*D_T.shape[1])) / (sum(HH_D**2)/(HH_D.shape[0]*HH_D.shape[1]))
    return ER

#定义函数去估计图像的噪音
def noise_estimation(HH_D, method):     #HH_D为HH子带系数矩阵,method为噪音的类型
    if method == "pepper_salt":
        noise_variance = np.median(HH_D) / 0.6745
    elif method == "gauss":
        p = HH_D.var()
        noise_variance = -1.3462 + 0.099995*p +0.00213*p*2 + 0.00002*p*p*p
    else:
        raise ("Type Error")
    return noise_variance


image_edge = edge_detection("../image/church.jpg")
image = cv2.imread("../image/church.jpg")
image = cv2.resize(image, (1080, 1080))
# 将多通道图像变为单通道图像
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = edge_process(image, image_edge)
image.astype(np.float32)


#添加噪音之后的图像列表
pep_salt_images = []
gauss_images = []

#噪音图像的HH子带系数
pep_salt_image_coeffs = []
gauss_image_coeffs = []
#迭代得到噪音图像及它们的HH子带系数
for i in range(0,12,2):
    pep_salt_image = add_noise(image, "pepper_salt", i)
    pep_salt_images.append(pep_salt_image)
    coeffs = pywt.dwt2(pep_salt_image, 'haar')
    cA, (cH, cV, cD) = coeffs           #其中cA为图像的LL系数，cH为LH系数，cV为HL系数以及cD为HH系数
    pep_salt_image_coeffs.append(cA)
    gauss_image = add_noise(image, "gauss", i)
    gauss_images.append(gauss_image)
    coeffs = pywt.dwt2(gauss_image, 'haar')
    cA, (cH, cV, cD) = coeffs           # 其中cA为图像的LL系数，cH为LH系数，cV为HL系数以及cD为HH系数
    gauss_image_coeffs.append(cA)

#噪音图像的HH子带系数的不同阈值下的能量比矩阵
pep_salt_ER = zeros((6, 9))
gauss_ER = zeros((6, 9))
#计算噪音图像的HH子带系数的不同阈值下的能量比矩阵
for i in range(0, 6):
    for j in range(100,1000,100):
        pep_salt_ER[i, j/100 -1] = insignificant_energy_ratio(pep_salt_image_coeffs[i], j)
        gauss_ER[i,j/100 -1] = insignificant_energy_ratio(gauss_image_coeffs[i], j)

#得出噪音图像的HH子带系数的不同阈值下的能量比图像
x = np.arange(100, 1000, 100)
plt.plot(x, pep_salt_ER[0], pep_salt_ER[1], pep_salt_ER[2], pep_salt_ER[3], pep_salt_ER[4], pep_salt_ER[5], color="r", linestyle="-", marker="*", linewidth=1)

plt.plot(x, gauss_ER[0], gauss_ER[1], gauss_ER[2], gauss_ER[3], gauss_ER[4], gauss_ER[5], color="k", linestyle="-", marker="^", linewidth=1)

plt.xlabel("T")
plt.ylabel("ER")
#不同噪音强度及阈值下的HH系数能量比
plt.title("Energy ratio of HH coefficient at different noise levels and thresholds")
plt.show()
