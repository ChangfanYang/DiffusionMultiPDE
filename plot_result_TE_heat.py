import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import torch

# 加载数据
TE_heat_results = sio.loadmat("/home/yangchangfan/CODE/DiffusionPDE/TE_heat_results.mat")
mater = TE_heat_results['mater']
complex_Ez = TE_heat_results['Ez']
real_Ez = complex_Ez.real
imag_Ez = complex_Ez.imag
T = TE_heat_results['T']

mater = np.squeeze(mater)
real_Ez = np.squeeze(real_Ez)
imag_Ez = np.squeeze(imag_Ez)
T = np.squeeze(T)

data_path = '/data/yangchangfan/DiffusionPDE/data/testing/TE_heat'
offset = 10001

# 定义文件路径
mater_GT = sio.loadmat(os.path.join(data_path, 'mater', f'{offset}.mat'))['mater']
complex_Ez_GT = sio.loadmat(os.path.join(data_path, 'Ez', f'{offset}.mat'))['export_Ez']
real_Ez_GT = complex_Ez_GT.real
imag_Ez_GT = complex_Ez_GT.imag
T_GT = sio.loadmat(os.path.join(data_path, 'T', f'{offset}.mat'))['export_T']


vmin_mater = mater_GT.min()
vmax_mater = mater_GT.max()
vmin_real_Ez = real_Ez_GT.min()
vmax_real_Ez = real_Ez_GT.max()
vmin_imag_Ez = imag_Ez_GT.min()
vmax_imag_Ez = imag_Ez_GT.max()
vmin_T = T_GT.min()
vmax_T = T_GT.max()

# 创建 2x4 的子图
fig, axes = plt.subplots(2, 4, figsize=(14, 7))

# 绘制第一行子图
im0 = axes[0, 0].imshow(mater, cmap='plasma', vmin=vmin_mater, vmax=vmax_mater)
axes[0, 0].set_title('mater')
axes[0, 0].axis('off')
plt.colorbar(im0, ax=axes[0, 0])  # 添加 colorbar

im1 = axes[0, 1].imshow(real_Ez, cmap='plasma', vmin=vmin_real_Ez, vmax=vmax_real_Ez)
axes[0, 1].set_title('real_Ez')
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1])  # 添加 colorbar

im2 = axes[0, 2].imshow(imag_Ez, cmap='plasma', vmin=vmin_imag_Ez, vmax=vmax_imag_Ez)
axes[0, 2].set_title('imag_Ez')
axes[0, 2].axis('off')
plt.colorbar(im2, ax=axes[0, 2])  # 添加 colorbar

im3 = axes[0, 3].imshow(T, cmap='plasma', vmin=vmin_T, vmax=vmax_T)
axes[0, 3].set_title('T')
axes[0, 3].axis('off')
plt.colorbar(im3, ax=axes[0, 3])  # 添加 colorbar

# 绘制第二行子图
im4 = axes[1, 0].imshow(mater_GT, cmap='plasma', vmin=vmin_mater, vmax=vmax_mater)
axes[1, 0].set_title('mater_GT')
axes[1, 0].axis('off')
plt.colorbar(im4, ax=axes[1, 0])  # 添加 colorbar

im5 = axes[1, 1].imshow(real_Ez_GT, cmap='plasma', vmin=vmin_real_Ez, vmax=vmax_real_Ez)
axes[1, 1].set_title('real_Ez_GT')
axes[1, 1].axis('off')
plt.colorbar(im5, ax=axes[1, 1])  # 添加 colorbar

im6 = axes[1, 2].imshow(imag_Ez_GT, cmap='plasma', vmin=vmin_imag_Ez, vmax=vmax_imag_Ez)
axes[1, 2].set_title('imag_Ez_GT')
axes[1, 2].axis('off')
plt.colorbar(im6, ax=axes[1, 2])  # 添加 colorbar

im7 = axes[1, 3].imshow(T_GT, cmap='plasma', vmin=vmin_T, vmax=vmax_T)
axes[1, 3].set_title('T_GT')
axes[1, 3].axis('off')
plt.colorbar(im7, ax=axes[1, 3])  # 添加 colorbar

# 调整子图间距
plt.tight_layout()

# 显示图像
plt.show()

# 保存图像
fig.suptitle('TE_heat Results', y=1.02)
plt.savefig('output_TE_heat.png')  # 保存为 PNG 文件


mater = torch.tensor(mater)
mater_GT = torch.tensor(mater_GT)
real_Ez = torch.tensor(real_Ez)
real_Ez_GT = torch.tensor(real_Ez_GT)
imag_Ez = torch.tensor(imag_Ez)
imag_Ez_GT = torch.tensor(imag_Ez_GT)
T = torch.tensor(T)
T_GT = torch.tensor(T_GT)

# 计算相对误差
relative_error_mater = torch.norm(mater - mater_GT, 2) / torch.norm(mater_GT, 2)
relative_error_real_Ez = torch.norm(real_Ez - real_Ez_GT, 2) / torch.norm(real_Ez_GT, 2)
relative_error_imag_Ez = torch.norm(imag_Ez - imag_Ez_GT, 2) / torch.norm(imag_Ez_GT, 2)
relative_error_T = torch.norm(T - T_GT, 2) / torch.norm(T_GT, 2)

print(f'Relative error of mater: {relative_error_mater}')
print(f'Relative error of real_Ez: {relative_error_real_Ez}')
print(f'Relative error of imag_Ez: {relative_error_imag_Ez}')
print(f'Relative error of T: {relative_error_T}')