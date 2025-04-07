import numpy as np

# 指定文件路径
file_path = "/data/yangchangfan/DiffusionPDE/data/NS_heat-merged/merge_1.npy"

# 加载.npy文件
data = np.load(file_path)

# 打印数据和形状
print("数据内容：")
print(data[:, :, 3])
print("\n数据形状：", data.shape)