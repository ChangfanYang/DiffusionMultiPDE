import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import torch

# 加载数据
NS_heat_results = sio.loadmat("/home/yangchangfan/CODE/DiffusionPDE/NS_heat_results.mat")
Q_heat = NS_heat_results['Q_heat']
u_u = NS_heat_results['u_u']
u_v = NS_heat_results['u_v']
T = NS_heat_results['T']

Q_heat = np.squeeze(Q_heat)
u_u = np.squeeze(u_u)
u_v = np.squeeze(u_v)
T = np.squeeze(T)

data_path = '/data/yangchangfan/DiffusionPDE/data/testing/NS_heat'
offset = 10001

# 定义文件路径
Q_heat_GT = sio.loadmat(os.path.join(data_path, 'Q_heat', f'{offset}.mat'))['export_Q_heat']
u_u_GT = sio.loadmat(os.path.join(data_path, 'u_u', f'{offset}.mat'))['export_u_u']
u_v_GT = sio.loadmat(os.path.join(data_path, 'u_v', f'{offset}.mat'))['export_u_v']
T_GT = sio.loadmat(os.path.join(data_path, 'T', f'{offset}.mat'))['export_T']


vmin_Q_heat = Q_heat_GT.min()
vmax_Q_heat = Q_heat_GT.max()
vmin_u_u = u_u_GT.min()
vmax_u_u = u_u_GT.max()
vmin_u_v = u_v_GT.min()
vmax_u_v = u_v_GT.max()
vmin_T = T_GT.min()
vmax_T = T_GT.max()

# 创建 2x4 的子图
fig, axes = plt.subplots(2, 4, figsize=(14, 7))

# 绘制第一行子图
im0 = axes[0, 0].imshow(Q_heat, cmap='plasma', vmin=vmin_Q_heat, vmax=vmax_Q_heat)
axes[0, 0].set_title('Q_heat')
axes[0, 0].axis('off')
plt.colorbar(im0, ax=axes[0, 0])  # 添加 colorbar

im1 = axes[0, 1].imshow(u_u, cmap='plasma', vmin=vmin_u_u, vmax=vmax_u_u)
axes[0, 1].set_title('u_u')
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1])  # 添加 colorbar

im2 = axes[0, 2].imshow(u_v, cmap='plasma', vmin=vmin_u_v, vmax=vmax_u_v)
axes[0, 2].set_title('u_v')
axes[0, 2].axis('off')
plt.colorbar(im2, ax=axes[0, 2])  # 添加 colorbar

im3 = axes[0, 3].imshow(T, cmap='plasma', vmin=vmin_T, vmax=vmax_T)
axes[0, 3].set_title('T')
axes[0, 3].axis('off')
plt.colorbar(im3, ax=axes[0, 3])  # 添加 colorbar

# 绘制第二行子图
im4 = axes[1, 0].imshow(Q_heat_GT, cmap='plasma', vmin=vmin_Q_heat, vmax=vmax_Q_heat)
axes[1, 0].set_title('Q_heat_GT')
axes[1, 0].axis('off')
plt.colorbar(im4, ax=axes[1, 0])  # 添加 colorbar

im5 = axes[1, 1].imshow(u_u_GT, cmap='plasma', vmin=vmin_u_u, vmax=vmax_u_u)
axes[1, 1].set_title('u_u_GT')
axes[1, 1].axis('off')
plt.colorbar(im5, ax=axes[1, 1])  # 添加 colorbar

im6 = axes[1, 2].imshow(u_v_GT, cmap='plasma', vmin=vmin_u_v, vmax=vmax_u_v)
axes[1, 2].set_title('u_v_GT')
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
fig.suptitle('NS_heat Results', y=1.02)
plt.savefig('output_NS.png')  # 保存为 PNG 文件


Q_heat = torch.tensor(Q_heat)
Q_heat_GT = torch.tensor(Q_heat_GT)
u_u = torch.tensor(u_u)
u_u_GT = torch.tensor(u_u_GT)
u_v = torch.tensor(u_v)
u_v_GT = torch.tensor(u_v_GT)
T = torch.tensor(T)
T_GT = torch.tensor(T_GT)

# 计算相对误差
relative_error_Q_heat = torch.norm(Q_heat - Q_heat_GT, 2) / torch.norm(Q_heat_GT, 2)
relative_error_u_u = torch.norm(u_u - u_u_GT, 2) / torch.norm(u_u_GT, 2)
relative_error_u_v = torch.norm(u_v - u_v_GT, 2) / torch.norm(u_v_GT, 2)
relative_error_T = torch.norm(T - T_GT, 2) / torch.norm(T_GT, 2)

print(f'Relative error of Q_heat: {relative_error_Q_heat}')
print(f'Relative error of u_u: {relative_error_u_u}')
print(f'Relative error of u_v: {relative_error_u_v}')
print(f'Relative error of T: {relative_error_T}')