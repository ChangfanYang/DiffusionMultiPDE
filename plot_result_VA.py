import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import torch

# 加载数据
VA_results = sio.loadmat("VA_results.mat")

rho_water = VA_results['rho_water']
p_t = VA_results['p_t']
Sxx = VA_results['Sxx']
Sxy = VA_results['Sxy']
Syy = VA_results['Syy']
x_u = VA_results['x_u']
x_v = VA_results['x_v']

rho_water = np.squeeze(rho_water)
p_t = np.squeeze(p_t)
Sxx = np.squeeze(Sxx)
Sxy = np.squeeze(Sxy)
Syy = np.squeeze(Syy)
x_u = np.squeeze(x_u)
x_v = np.squeeze(x_v)

data_path = '/data/yangchangfan/DiffusionPDE/data/testing/VA'
offset = 10001

# 定义文件路径
rho_water_GT = sio.loadmat(os.path.join(data_path, 'rho_water', f'{offset}.mat'))['export_rho_water']
p_t_GT = sio.loadmat(os.path.join(data_path, 'p_t', f'{offset}.mat'))['export_p_t']
Sxx_GT = sio.loadmat(os.path.join(data_path, 'Sxx', f'{offset}.mat'))['export_Sxx']
Sxy_GT = sio.loadmat(os.path.join(data_path, 'Sxy', f'{offset}.mat'))['export_Sxy']
Syy_GT = sio.loadmat(os.path.join(data_path, 'Syy', f'{offset}.mat'))['export_Syy']
x_u_GT = sio.loadmat(os.path.join(data_path, 'x_u', f'{offset}.mat'))['export_x_u']
x_v_GT = sio.loadmat(os.path.join(data_path, 'x_v', f'{offset}.mat'))['export_x_v']


# 计算每个变量的最小值和最大值
# vmin_kappa, vmax_kappa = kappa_GT.min(), kappa_GT.max()
# vmin_ec_V, vmax_ec_V = ec_V_GT.min(), ec_V_GT.max()

# vmin_u_flow, vmax_u_flow = u_flow_GT.min(), u_flow_GT.max()
# vmin_v_flow, vmax_v_flow = v_flow_GT.min(), v_flow_GT.max()

# 创建 2x6 的子图
fig, axes = plt.subplots(2, 7, figsize=(18, 8))



# 定义要绘制的变量及其标题
variables = {
    'rho_water': (rho_water, rho_water_GT),
    'p_t': (p_t, p_t_GT),
    'Sxx': (Sxx, Sxx_GT),
    'Sxy': (Sxy, Sxy_GT),
    'Syy': (Syy, Syy_GT),
    'x_u': (x_u, x_u_GT),
    'x_v': (x_v, x_v_GT)
    
}

# 绘制图像
for col, (name, (data, gt_data)) in enumerate(variables.items()):

    vmin = gt_data.real.min()
    vmax = gt_data.real.max()

    # 绘制预测结果
    im_pred = axes[0, col].imshow(data.real, cmap='inferno',vmin=vmin,vmax=vmax)  #
    axes[0, col].set_title(f'{name}')
    axes[0, col].axis('off')
    plt.colorbar(im_pred, ax=axes[0, col])

    # 绘制真实结果
    im_gt = axes[1, col].imshow(gt_data.real, cmap='inferno',vmin=vmin,vmax=vmax)  #
    axes[1, col].set_title(f'{name}_GT')
    axes[1, col].axis('off')
    plt.colorbar(im_gt, ax=axes[1, col])

# 调整子图间距
plt.tight_layout()

# 显示图像
plt.show()

# 保存图像
fig.suptitle('VA Results', y=1.02)
plt.savefig('output_VA.png')  # 保存为 PNG 文件



# 转换为 PyTorch 张量
rho_water = torch.tensor(rho_water)
rho_water_GT = torch.tensor(rho_water_GT)
p_t = torch.tensor(p_t)
p_t_GT = torch.tensor(p_t_GT)
Sxx = torch.tensor(Sxx)
Sxx_GT = torch.tensor(Sxx_GT)
Sxy = torch.tensor(Sxy)
Sxy_GT = torch.tensor(Sxy_GT)
Syy = torch.tensor(Syy)
Syy_GT = torch.tensor(Syy_GT)
x_u = torch.tensor(x_u)
x_u_GT = torch.tensor(x_u_GT)
x_v = torch.tensor(x_v)
x_v_GT = torch.tensor(x_v_GT)

# 计算相对误差
def calculate_relative_error(pred, gt):
    return torch.norm(pred - gt, 2) / torch.norm(gt, 2)

relative_error_rho_water = calculate_relative_error(rho_water, rho_water_GT)
relative_error_p_t = calculate_relative_error(p_t, p_t_GT)
relative_error_Sxx = calculate_relative_error(Sxx, Sxx_GT)
relative_error_Sxy = calculate_relative_error(Sxy, Sxy_GT)
relative_error_Syy = calculate_relative_error(Syy, Syy_GT)
relative_error_x_u = calculate_relative_error(x_u, x_u_GT)
relative_error_x_v = calculate_relative_error(x_v, x_v_GT)

# 打印相对误差
print(f'Relative error of rho_water: {relative_error_rho_water.item():.6e}')
print(f'Relative error of p_t: {relative_error_p_t.item():.6e}')
print(f'Relative error of Sxx: {relative_error_Sxx.item():.6e}')
print(f'Relative error of Sxy: {relative_error_Sxy.item():.6e}')
print(f'Relative error of Syy: {relative_error_Syy.item():.6e}')
print(f'Relative error of x_u: {relative_error_x_u.item():.6e}')
print(f'Relative error of x_v: {relative_error_x_v.item():.6e}')