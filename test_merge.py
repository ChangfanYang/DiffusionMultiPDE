import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io
import os

def load_ranges(base_path, variables):
    ranges = {}
    
    # rho_water
    rho_data = scipy.io.loadmat(f"{base_path}/rho_water/range_allrho_water.mat")['range_allrho_water']
    ranges['rho_water'] = {'max': rho_data[0, 1], 'min': rho_data[0, 0]}
    
    # 加载其他变量的范围
    for var in variables:
        real_data = scipy.io.loadmat(f"{base_path}/{var}/range_allreal_{var}.mat")[f'range_allreal_{var}']
        imag_data = scipy.io.loadmat(f"{base_path}/{var}/range_allimag_{var}.mat")[f'range_allimag_{var}']
        
        ranges[var] = {
            'max_real': real_data[0, 1],
            'min_real': real_data[0, 0],
            'max_imag': imag_data[0, 1],
            'min_imag': imag_data[0, 0]
        }
    
    return ranges

# 加载数据
data = np.load("/data/yangchangfan/DiffusionPDE/data/VA_wrong-merged/merge_2.npy")
slice_9 = data[:, :, 9]
slice_11 = data[:, :, 11]

slice_10 = data[:, :, 10]
slice_12 = data[:, :, 12]

base_path = "/data/yangchangfan/DiffusionPDE/data/training/VA"
variables = ['p_t', 'Sxx', 'Sxy', 'Syy', 'x_u', 'x_v']

data_path = "/data/yangchangfan/DiffusionPDE/data/training/VA"
range_data = load_ranges(base_path, variables)


# print(range_data['x_u']['max_real'])
# print(range_data['x_u']['min_real'])


# 数据归一化
# slice_9 = ((slice_9 + 0.9) / 1.8 * (range_data['x_u']['max_real'] - range_data['x_u']['min_real']) + range_data['x_u']['min_real'])
# slice_11 = ((slice_11 + 0.9) / 1.8 * (range_data['x_v']['max_real'] - range_data['x_v']['min_real']) + range_data['x_v']['min_real'])


# slice_10 = ((slice_10 + 0.9) / 1.8 * (range_data['x_u']['max_imag'] - range_data['x_u']['min_imag']) + range_data['x_u']['min_imag'])
# slice_12 = ((slice_12 + 0.9) / 1.8 * (range_data['x_v']['max_imag'] - range_data['x_v']['min_imag']) + range_data['x_v']['min_imag'])

# # 加载 GT 数据
# omega = np.pi*1e5
# rho_AL=2730

# x_u_GT = scipy.io.loadmat(os.path.join(data_path, 'x_u', f'{2}.mat'))['export_x_u']
# x_v_GT = scipy.io.loadmat(os.path.join(data_path, 'x_v', f'{2}.mat'))['export_x_v']
# x_u_GT = x_u_GT /(omega**2 * rho_AL)
# x_v_GT = x_v_GT /(omega**2 * rho_AL)

# vmin_x_u = x_u_GT.real.min()
# vmax_x_u = x_u_GT.real.max()

# vmin_x_v = x_v_GT.real.min()
# vmax_x_v = x_v_GT.real.max()

# # 创建画布
# plt.figure(figsize=(20, 10))

# # 第一张子图（data[:, :, 9]）
# plt.subplot(2, 2, 1)  # 2行2列，第1个位置 ,vmin=vmin_x_u,vmax=vmax_x_u
# plt.imshow(slice_9, cmap='inferno')
# plt.colorbar()
# plt.title("Slice at index 9")
# plt.axis('off')

# # 第二张子图（data[:, :, 11]）
# plt.subplot(2, 2, 2)  # 2行2列，第2个位置  ,vmin=vmin_x_v,vmax=vmax_x_v
# plt.imshow(slice_11, cmap='inferno')
# plt.colorbar()
# plt.title("Slice at index 11")
# plt.axis('off')

# # 第三张子图（x_u_GT）
# plt.subplot(2, 2, 3)  # 2行2列，第3个位置
# plt.imshow(x_u_GT.real, cmap='inferno',vmin=vmin_x_u,vmax=vmax_x_u)
# plt.colorbar()
# plt.title("GT at index 9")
# plt.axis('off')

# # 第四张子图（x_v_GT）
# plt.subplot(2, 2, 4)  # 2行2列，第4个位置
# plt.imshow(x_v_GT.real, cmap='inferno',vmin=vmin_x_v,vmax=vmax_x_v)
# plt.colorbar()
# plt.title("GT at index 11")
# plt.axis('off')

# # 保存图片
# plt.tight_layout()
# output_path = "/home/yangchangfan/CODE/DiffusionPDE/test_merge_VA_x_v_imag.png"
# plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
# plt.close()


# print("Br Min:", Br_normalized.min(), "Max:", Br_normalized.max())
# print("Jx Min:", Jx_normalized.min(), "Max:", Jx_normalized.max())
# print("Jy Min:", Jy_normalized.min(), "Max:", Jy_normalized.max())
# print("Jz Min:", Jz_normalized.min(), "Max:", Jz_normalized.max())
# print("u_u Min:", u_u_normalized.min(), "Max:", u_u_normalized.max())
# print("u_v Min:", u_v_normalized.min(), "Max:", u_v_normalized.max())



# import scipy.io as sio

# file_path = "/data/yangchangfan/DiffusionPDE/data/training/ns-nonbounded/ns-nonbounded_8.mat"
# data = sio.loadmat(file_path)

# # 查看字典的键
# print("Keys in the MATLAB file:", data.keys())

# # 提取具体的数组
# a = data['a']
# t = data['t']
# u = data['u']

# # 查看数组的形状
# print("Shape of 'a':", a.shape)
# print("Shape of 't':", t.shape)
# print("Shape of 'u':", u.shape)

# # 打印数组的内容（可选）
# # print("Content of 'a':", a)
# # print("Content of 't':", t)
# # print("Content of 'u':", u)



# import numpy as np

# file_path = "/data/yangchangfan/DiffusionPDE/data/TE_heat-merged/merge_1.npy"
# data = np.load(file_path)

# # Check the shape and content
# print("Array shape:", data.shape)
# print("Data type:", data[:,:,3])
# print("Min value:", np.min(data[:,:,3]))
# print("Max value:", np.max(data[:,:,3]))



# 打印数组的内容（可选）
# print("Content of 'a':", a)
# print("Content of 't':", t)
# print("Content of 'u':", u)


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
