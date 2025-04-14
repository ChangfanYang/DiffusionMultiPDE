import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import torch

# 加载数据
E_flow_results = sio.loadmat("/home/yangchangfan/CODE/DiffusionPDE/E_flow_results.mat")
kappa = E_flow_results['kappa']
ec_V = E_flow_results['ec_V']
u_flow = E_flow_results['u_flow']
v_flow = E_flow_results['v_flow']

kappa = np.squeeze(kappa)
ec_V = np.squeeze(ec_V)

u_flow = np.squeeze(u_flow)
v_flow = np.squeeze(v_flow)

data_path = '/data/yangchangfan/DiffusionPDE/data/testing/E_flow'
offset = 10001

# 定义文件路径
kappa_GT = sio.loadmat(os.path.join(data_path, 'kappa', f'{offset}.mat'))['export_kappa']
ec_V_GT = sio.loadmat(os.path.join(data_path, 'ec_V', f'{offset}.mat'))['export_ec_V']
u_flow_GT = sio.loadmat(os.path.join(data_path, 'u_flow', f'{offset}.mat'))['export_u_flow']
v_flow_GT = sio.loadmat(os.path.join(data_path, 'v_flow', f'{offset}.mat'))['export_v_flow']

# 计算每个变量的最小值和最大值
vmin_kappa, vmax_kappa = kappa_GT.min(), kappa_GT.max()
vmin_ec_V, vmax_ec_V = ec_V_GT.min(), ec_V_GT.max()

vmin_u_flow, vmax_u_flow = u_flow_GT.min(), u_flow_GT.max()
vmin_v_flow, vmax_v_flow = v_flow_GT.min(), v_flow_GT.max()

# 创建 2x6 的子图
fig, axes = plt.subplots(2, 4, figsize=(18, 8))

# 定义要绘制的变量及其标题
variables = {
    'kappa': (kappa, kappa_GT, vmin_kappa, vmax_kappa),
    'ec_V': (ec_V, ec_V_GT, vmin_ec_V, vmax_ec_V),

    'u_flow': (u_flow, u_flow_GT, vmin_u_flow, vmax_u_flow),
    'v_flow': (v_flow, v_flow_GT, vmin_v_flow, vmax_v_flow)
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
fig.suptitle('E_flow Results', y=1.02)
plt.savefig('output_E_flow.png')  # 保存为 PNG 文件

# 转换为 PyTorch 张量
kappa = torch.tensor(kappa)
kappa_GT = torch.tensor(kappa_GT)
ec_V = torch.tensor(ec_V)
ec_V_GT = torch.tensor(ec_V_GT)
u_flow = torch.tensor(u_flow)
u_flow_GT = torch.tensor(u_flow_GT)
v_flow = torch.tensor(v_flow)
v_flow_GT = torch.tensor(v_flow_GT)

# 计算相对误差
def calculate_relative_error(pred, gt):
    return torch.norm(pred - gt, 2) / torch.norm(gt, 2)

relative_error_kappa = calculate_relative_error(kappa, kappa_GT)
relative_error_ec_V = calculate_relative_error(ec_V, ec_V_GT)
relative_error_u_flow = calculate_relative_error(u_flow, u_flow_GT)
relative_error_v_flow = calculate_relative_error(v_flow, v_flow_GT)

# 打印相对误差
print(f'Relative error of kappa: {relative_error_kappa.item():.6e}')
print(f'Relative error of ec_V: {relative_error_ec_V.item():.6e}')
print(f'Relative error of u_flow: {relative_error_u_flow.item():.6e}')
print(f'Relative error of v_flow: {relative_error_v_flow.item():.6e}')