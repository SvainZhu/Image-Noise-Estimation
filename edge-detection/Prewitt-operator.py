# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('../image/church.jpg')
# 首先将原图像进行边界扩展，并将其转换为灰度图
image = cv2.resize(image, (1080, 1080))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Prewitt算子
operator_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
operator_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
x = cv2.filter2D(image, cv2.CV_16S, operator_x)
y = cv2.filter2D(image, cv2.CV_16S, operator_y)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

# 生成经Prewitt算子处理之后的边缘图像
Prewitt_image = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
Prewitt_image = 255 - Prewitt_image

# 显示图像
plt.imshow(Prewitt_image, cmap="gray")
plt.axis("off")
plt.savefig("Edge image by Perwitt operator.jpg")