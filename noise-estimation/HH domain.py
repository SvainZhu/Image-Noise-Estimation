# -*- coding:utf8 -*-
import cv2
from pylab import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as signal


# 生成高斯算子的函数
def func(x, y, sigma=1):
    return 100 * (1 / (2 * np.pi * sigma)) * np.exp(-((x - 2) ** 2 + (y - 2) ** 2) / (2.0 * sigma ** 2))

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

def edge_process(image1,image2):    #image1为原始图像，image2为边缘图像
    for i in range(1, image1.shape[0] - 1):
        for j in range(1, image1.shape[1] - 1):
            if image2[i,j] != 255:
                image1[i,j] = mean(image1[i,j-1],image1[i,j+1],image1[i-1,j],image1[i+1,j])
    return image1



