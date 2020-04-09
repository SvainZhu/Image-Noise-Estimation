# -*- coding:utf8 -*-
import cv2
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import pywt
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
image = cv2.imread("../image/church.jpg")
image = cv2.resize(image, (1080, 1080))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image.astype(np.float32)

#生成椒盐噪音强度为25%的噪音图像并进行小波变换
pep_salt_image = add_noise(image, "pepper_salt", 25)
cv2.imwrite("./pep_salt_image.jpg", pep_salt_image)
coeffs = pywt.dwt2(pep_salt_image, 'haar')
cA, (cH, cV, cD) = coeffs           # 其中cA为图像的LL系数，cH为LH系数，cV为HL系数以及cD为HH系数
HH_salt_image = cD.astype(np.int64)

#统计HH子带小波系数的分布情况
amp_range = 300
Pcoeff_salt_image = zeros((2*amp_range, 1))
for i in HH_salt_image:
    for j in i:
        Pcoeff_salt_image[j+amp_range-1, 0] += 1


#生成高斯白噪音强度为25的噪音图像并进行小波变换
gauss_image = add_noise(image, "gauss", 25)
cv2.imwrite("./gauss_image.jpg", gauss_image)
coeffs = pywt.dwt2(gauss_image, 'haar')
cA, (cH, cV, cD) = coeffs           # 其中cA为图像的LL系数，cH为LH系数，cV为HL系数以及cD为HH系数
HH_gauss_image = cD.astype(np.int64)
#统计HH子带小波系数的分布情况
Pcoeff_gauss_image = zeros((2*amp_range, 1))
for i in HH_gauss_image:
    for j in i:
        Pcoeff_gauss_image[j+amp_range-1, 0] += 1

#画出椒盐噪音图像下噪音图像的HH子带小波系数不同幅值系数出现的概率分布图并保存
x = np.arange(-amp_range, amp_range, 1)
plt.plot(x, Pcoeff_salt_image, color="k")
plt.xlabel("Coe")
plt.ylabel("Pcoe")
plt.title("Amplitude distribution diagram of HH wavelet coefficient about Pepper-Salt noise image")
plt.savefig("pep-salt_HH_distribution.jpg")
plt.clf()

#画出高斯噪音图像下噪音图像的HH子带小波系数不同幅值系数出现的概率分布图并保存
plt.plot(x, Pcoeff_gauss_image, color="k")
plt.xlabel("Coe")
plt.ylabel("Pcoe")
plt.title("Amplitude distribution diagram of HH wavelet coefficient about Gauss noise image")
plt.savefig("gauss_HH_distribution.jpg")
plt.clf()