import numpy as np

# 指定文件路径
file_path = "/data/yangchangfan/DiffusionPDE/data/MHD-merged//merge_1000.npy"

# 加载.npy文件
data = np.load(file_path)

# 打印数据和形状
print("数据内容：")
print(data[:, :, 3])
print("\n数据形状：", data.shape)


print("Min:", data.min(), "Max:", data.max())
# print("Br Min:", Br_normalized.min(), "Max:", Br_normalized.max())
# print("Jx Min:", Jx_normalized.min(), "Max:", Jx_normalized.max())
# print("Jy Min:", Jy_normalized.min(), "Max:", Jy_normalized.max())
# print("Jz Min:", Jz_normalized.min(), "Max:", Jz_normalized.max())
# print("u_u Min:", u_u_normalized.min(), "Max:", u_u_normalized.max())
# print("u_v Min:", u_v_normalized.min(), "Max:", u_v_normalized.max())