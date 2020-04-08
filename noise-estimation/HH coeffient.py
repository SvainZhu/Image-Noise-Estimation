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



image = cv2.imread("../image/church.jpg")
image = cv2.resize(image, (1080, 1080))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image.astype(np.float32)
pep_salt_image = add_noise(image, "pepper_salt", 40)
coeffs = pywt.dwt2(pep_salt_image, 'haar')
cA, (cH, cV, cD) = coeffs           # 其中cA为图像的LL系数，cH为LH系数，cV为HL系数以及cD为HH系数
HH_salt_image = cD.astype(np.int64)
#HH_salt_image.astype(np.int64)
Pcoeff_salt_image = zeros((400,1))
for i in HH_salt_image:
    for j in i:
        Pcoeff_salt_image[j+199, 0] += 1


gauss_image = add_noise(image, "gauss", 40)
coeffs = pywt.dwt2(gauss_image, 'haar')
cA, (cH, cV, cD) = coeffs           # 其中cA为图像的LL系数，cH为LH系数，cV为HL系数以及cD为HH系数
HH_gauss_image = cD.astype(np.int64)
#HH_gauss_image.astype(np.int32)
Pcoeff_gauss_image = zeros((400,1))
for i in HH_gauss_image:
    for j in i:
        Pcoeff_gauss_image[j+199, 0] += 1

plt.imshow(pep_salt_image, cmap=cm.gray)
plt.axis("off")
plt.show()

plt.imshow(gauss_image, cmap=cm.gray)
plt.axis("off")
plt.show()
#得出不同噪音图像下噪音图像的HH子带小波系数不同幅值系数出现的概率
x = np.arange(-200, 200, 1)
plt.plot(x, Pcoeff_salt_image,color="k")
plt.xlabel("N")
plt.ylabel("coeffient")
#plt.title("Energy ratio of HH coefficient at different noise levels and thresholds")
plt.show()
