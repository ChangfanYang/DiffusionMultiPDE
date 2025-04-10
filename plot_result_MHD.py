import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import torch

# 加载数据
MHD_results = sio.loadmat("/home/yangchangfan/CODE/DiffusionPDE/MHD_results.mat")
Br = MHD_results['Br']
Jx = MHD_results['Jx']
Jy = MHD_results['Jy']
Jz = MHD_results['Jz']
u_u = MHD_results['u_u']
u_v = MHD_results['u_v']

Br = np.squeeze(Br)
Jx = np.squeeze(Jx)
Jy = np.squeeze(Jy)
Jz = np.squeeze(Jz)
u_u = np.squeeze(u_u)
u_v = np.squeeze(u_v)

data_path = '/data/yangchangfan/DiffusionPDE/data/testing/MHD'
offset = 10001

# 定义文件路径
Br_GT = sio.loadmat(os.path.join(data_path, 'Br', f'{offset}.mat'))['export_Br']
Jx_GT = sio.loadmat(os.path.join(data_path, 'Jx', f'{offset}.mat'))['export_Jx']
Jy_GT = sio.loadmat(os.path.join(data_path, 'Jy', f'{offset}.mat'))['export_Jy']
Jz_GT = sio.loadmat(os.path.join(data_path, 'Jz', f'{offset}.mat'))['export_Jz']
u_u_GT = sio.loadmat(os.path.join(data_path, 'u_u', f'{offset}.mat'))['export_u']
u_v_GT = sio.loadmat(os.path.join(data_path, 'u_v', f'{offset}.mat'))['export_v']

# 计算每个变量的最小值和最大值
vmin_Br, vmax_Br = Br_GT.min(), Br_GT.max()
vmin_Jx, vmax_Jx = Jx_GT.min(), Jx_GT.max()
vmin_Jy, vmax_Jy = Jy_GT.min(), Jy_GT.max()
vmin_Jz, vmax_Jz = Jz_GT.min(), Jz_GT.max()
vmin_u_u, vmax_u_u = u_u_GT.min(), u_u_GT.max()
vmin_u_v, vmax_u_v = u_v_GT.min(), u_v_GT.max()

# 创建 2x6 的子图
fig, axes = plt.subplots(2, 6, figsize=(18, 8))

# 定义要绘制的变量及其标题
variables = {
    'Br': (Br, Br_GT, vmin_Br, vmax_Br),
    'Jx': (Jx, Jx_GT, vmin_Jx, vmax_Jx),
    'Jy': (Jy, Jy_GT, vmin_Jy, vmax_Jy),
    'Jz': (Jz, Jz_GT, vmin_Jz, vmax_Jz),
    'u_u': (u_u, u_u_GT, vmin_u_u, vmax_u_u),
    'u_v': (u_v, u_v_GT, vmin_u_v, vmax_u_v)
}

# 绘制图像
for col, (name, (data, gt_data, vmin, vmax)) in enumerate(variables.items()):
    # 绘制预测结果
    im_pred = axes[0, col].imshow(data, cmap='inferno', vmin=vmin, vmax=vmax)
    axes[0, col].set_title(f'{name}')
    axes[0, col].axis('off')
    plt.colorbar(im_pred, ax=axes[0, col])

    # 绘制真实结果
    im_gt = axes[1, col].imshow(gt_data, cmap='inferno', vmin=vmin, vmax=vmax)
    axes[1, col].set_title(f'{name}_GT')
    axes[1, col].axis('off')
    plt.colorbar(im_gt, ax=axes[1, col])

# 调整子图间距
plt.tight_layout()

# 显示图像
plt.show()

# 保存图像
fig.suptitle('MHD Results', y=1.02)
plt.savefig('output_MHD.png')  # 保存为 PNG 文件

# 转换为 PyTorch 张量
Br = torch.tensor(Br)
Br_GT = torch.tensor(Br_GT)
Jx = torch.tensor(Jx)
Jx_GT = torch.tensor(Jx_GT)
Jy = torch.tensor(Jy)
Jy_GT = torch.tensor(Jy_GT)
Jz = torch.tensor(Jz)
Jz_GT = torch.tensor(Jz_GT)
u_u = torch.tensor(u_u)
u_u_GT = torch.tensor(u_u_GT)
u_v = torch.tensor(u_v)
u_v_GT = torch.tensor(u_v_GT)

# 计算相对误差
def calculate_relative_error(pred, gt):
    return torch.norm(pred - gt, 2) / torch.norm(gt, 2)

relative_error_Br = calculate_relative_error(Br, Br_GT)
relative_error_Jx = calculate_relative_error(Jx, Jx_GT)
relative_error_Jy = calculate_relative_error(Jy, Jy_GT)
relative_error_Jz = calculate_relative_error(Jz, Jz_GT)
relative_error_u_u = calculate_relative_error(u_u, u_u_GT)
relative_error_u_v = calculate_relative_error(u_v, u_v_GT)

# 打印相对误差
print(f'Relative error of Br: {relative_error_Br.item():.6e}')
print(f'Relative error of Jx: {relative_error_Jx.item():.6e}')
print(f'Relative error of Jy: {relative_error_Jy.item():.6e}')
print(f'Relative error of Jz: {relative_error_Jz.item():.6e}')
print(f'Relative error of u_u: {relative_error_u_u.item():.6e}')
print(f'Relative error of u_v: {relative_error_u_v.item():.6e}')