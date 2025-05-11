import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os


data_Elder = np.load("/data/yangchangfan/DiffusionPDE/data/Elder_wrong-merged/merge_2.npy")

print(data_Elder.shape)
# exit()


plt.figure(figsize=(12, 8))  # 可以调整窗口大小
channels = [0, 1, 12, 23]
# channels = [0, 2, 13, 24]
# channels = [0, 3, 14, 25]
# channels = [0, 4, 15, 26]
# channels = [0, 5, 16, 27]
# channels = [0, 6, 17, 28]
# channels = [0, 7, 18, 29]
# channels = [0, 8, 19, 30]
# channels = [0, 9, 20, 31]
channels = [0, 10, 21, 32]
# channels = [0, 11, 22, 33]

# 绘制子图
for i, channel in enumerate(channels):
    plt.subplot(1, 4, i + 1)  # 创建 2x2 的子图布局
    plt.imshow(data_Elder[:, :, channel])  # 绘制第 i 个通道的数据
    plt.title(f"Channel {i}")  # 添加标题
    plt.colorbar()  # 添加颜色条（可选）

# 保存图像
plt.savefig("merge_Elder.png")
