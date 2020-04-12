# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('../image/church.jpg')
# 首先将原图像进行边界扩展，并将其转换为灰度图
image = cv2.resize(image, (1080, 1080))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Sobel算子
x = cv2.Sobel(image, cv2.CV_16S, 1, 0)  # 对x求一阶导
y = cv2.Sobel(image, cv2.CV_16S, 0, 1)  # 对y求一阶导
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

# 生成经Sobel算子处理之后的边缘图像
Sobel_image = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
Sobel_image = 255 -Sobel_image

# 显示图像
plt.imshow(Sobel_image, cmap="gray")
plt.axis("off")
plt.savefig("Edge image by Sobel operator.jpg")
