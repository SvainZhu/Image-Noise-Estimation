# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
import cv2

image = plt.imread('../image/church.jpg')
# 将原图像进行边界扩展，并将其转换为灰度图
image = cv2.resize(image, (1080, 1080))
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

sigma1 = sigma2 = 1
sum = 0

# 生成二维高斯分布矩阵
gaussian = np.zeros([5, 5])
for i in range(5):
    for j in range(5):
        gaussian[i, j] = math.exp(-1 / 2 * (np.square(i - 3) / np.square(sigma1) + (np.square(j - 3) / np.square(sigma2)))) / (2 * math.pi * sigma1 * sigma2)
        sum = sum + gaussian[i, j]
gaussian = gaussian / sum

# 高斯滤波
W, H = image.shape
new_image = np.zeros([W - 5, H - 5])
for i in range(W - 5):
    for j in range(H - 5):
        new_image[i, j] = np.sum(image[i:i + 5, j:j + 5] * gaussian)  # 与高斯矩阵卷积实现滤波

# 通过求梯度幅值
W1, H1 = new_image.shape
dx = np.zeros([W1 - 1, H1 - 1])
dy = np.zeros([W1 - 1, H1 - 1])
d = np.zeros([W1 - 1, H1 - 1])
for i in range(W1 - 1):
    for j in range(H1 - 1):
        dx[i, j] = new_image[i, j + 1] - new_image[i, j]
        dy[i, j] = new_image[i + 1, j] - new_image[i, j]
        d[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))  # 原图像梯度值作为梯度图像强度值


# 非极大值抑制
W2, H2 = d.shape
NMS = np.copy(d)
NMS[0, :] = NMS[W2 - 1, :] = NMS[:, 0] = NMS[:, H2 - 1] = 0
for i in range(1, W2 - 1):
    for j in range(1, H2 - 1):
        if d[i, j] == 0:
            NMS[i, j] = 0
        else:
            gradX = dx[i, j]
            gradY = dy[i, j]
            gradTemp = d[i, j]
            # 如果Y方向幅度值较大
            if np.abs(gradY) > np.abs(gradX):
                weight = np.abs(gradX) / np.abs(gradY)
                grad2 = d[i - 1, j]
                grad4 = d[i + 1, j]
                # 如果x,y方向梯度符号相同
                if gradX * gradY > 0:
                    grad1 = d[i - 1, j - 1]
                    grad3 = d[i + 1, j + 1]
                # 如果x,y方向梯度符号相反
                else:
                    grad1 = d[i - 1, j + 1]
                    grad3 = d[i + 1, j - 1]
            # 如果X方向幅度值较大
            else:
                weight = np.abs(gradY) / np.abs(gradX)
                grad2 = d[i, j - 1]
                grad4 = d[i, j + 1]
                # 如果x,y方向梯度符号相同
                if gradX * gradY > 0:
                    grad1 = d[i + 1, j - 1]
                    grad3 = d[i - 1, j + 1]
                # 如果x,y方向梯度符号相反
                else:
                    grad1 = d[i - 1, j - 1]
                    grad3 = d[i + 1, j + 1]
            gradTemp1 = weight * grad1 + (1 - weight) * grad2
            gradTemp2 = weight * grad3 + (1 - weight) * grad4
            if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                NMS[i, j] = gradTemp
            else:
                NMS[i, j] = 0

# 双阈值算法检测及连接边缘
W3, H3 = NMS.shape
DT = np.zeros([W3, H3])
TL = 0.1 * np.max(NMS)  #定义低阈值
TH = 0.3 * np.max(NMS)  #定义高阈值
for i in range(1, W3 - 1):
    for j in range(1, H3 - 1):
        if (NMS[i, j] < TL):
            DT[i, j] = 0
        elif (NMS[i, j] > TH):
            DT[i, j] = 1
        elif ((NMS[i - 1, j - 1:j + 1] < TH).any() or (NMS[i + 1, j - 1:j + 1]).any()
              or (NMS[i, [j - 1, j + 1]] < TH).any()):
            DT[i, j] = 1

#反转图像便于观察
dst = 255 - DT
plt.imshow(dst, cmap=cm.gray)
plt.axis("off")
plt.savefig("Edge image by Canny operator.jpg")