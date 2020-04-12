# -*- coding:utf8 -*-
import cv2
from pylab import *

image = cv2.imread("../image/church.jpg")
# 将原图像进行边界扩展，并将其转换为灰度图
image = cv2.resize(image, (1080, 1080))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def RobertsOperator(roi):
    operator_x = np.array([[-1, 0], [0, 1]])
    operator_y = np.array([[0, -1], [1, 0]])
    return np.abs(np.sum(roi[1:, 1:] * operator_x)) + np.abs(np.sum(roi[1:, 1:] * operator_y))


def RobertsAlogrithm(image):
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    for i in range(1, image.shape[0]):
        for j in range(1, image.shape[1]):
            image[i, j] = RobertsOperator(image[i - 1:i + 2, j - 1:j + 2])
    return image[1:image.shape[0], 1:image.shape[1]]


Roberts_image = RobertsAlogrithm(image)
plt.imshow(Roberts_image, cmap="binary")
plt.axis("off")
plt.savefig("Edge image by Roberts operator")