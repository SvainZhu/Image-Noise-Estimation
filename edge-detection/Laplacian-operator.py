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

# 生成标准差为5的5*5高斯算子
operator_5 = np.fromfunction(func, (5, 5), sigma=5)

# Laplace扩展算子
operator_3 = np.array([[1, 1, 1],
                    [1, -8, 1],
                    [1, 1, 1]])

# 读取图像
image = cv2.imread('../image/church.jpg')
# 首先将原图像进行边界扩展，并将其转换为灰度图
image = cv2.resize(image, (1080, 1080))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 利用生成的高斯算子与原图像进行卷积对图像进行平滑处理
image_blur = signal.convolve2d(image, operator_5, mode="same")

# 对平滑后的图像进行边缘检测
edge_image = signal.convolve2d(image_blur, operator_3, mode="same")

# 结果转化到0-255
edge_image = (edge_image / float(edge_image.max())) * 255

# 将大于灰度平均值的灰度值变成255（白色），便于观察边缘
edge_image[edge_image > edge_image.mean()] = 255

# 显示图像
plt.imshow(edge_image, cmap=cm.gray)
plt.axis("off")
plt.savefig("Edge image by improved algorithm.jpg")
plt.clf()
plt.imshow(image, cmap=cm.gray)
plt.axis("off")
plt.savefig("church.jpg")