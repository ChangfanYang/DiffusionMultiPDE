import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
Elder_results = sio.loadmat("/home/yangchangfan/CODE/DiffusionPDE/Elder_results.mat")
S_c = Elder_results['S_c']
if S_c.ndim == 11:
    S_c = np.expand_dims(S_c, axis=0)
S_c = np.repeat(S_c, 11, axis=0)  # shape: [11, 128, 128]
u_u = Elder_results['u_u']
u_v = Elder_results['u_v']
c_flow = Elder_results['c_flow']

S_c = np.squeeze(S_c)
u_u = np.squeeze(u_u)
u_v = np.squeeze(u_v)
c_flow = np.squeeze(c_flow)

data_test_path = "/data/yangchangfan/DiffusionPDE/data/testing/Elder/"
offset = 1001
time_steps = 11
C, H, W = 34, 128, 128

combined_data_GT = np.zeros((C, H, W), dtype=np.float64)

# ---------- 读取 S_c ----------
path_Sc = os.path.join(data_test_path, 'S_c', str(offset), '0.mat')
Sc_data = loadmat(path_Sc)
Sc = list(Sc_data.values())[-1]
combined_data_GT[0, :, :] = Sc

# ---------- 读取初始场 + 时域场 ----------
var_names = ['u_u', 'u_v', 'c_flow']
for var_idx, var in enumerate(var_names):

    # 时域场 t=0~10（通道 4 ~ 33）
    for t in range(0, time_steps):
        path_t = os.path.join(data_test_path, var, str(offset), f'{t}.mat')
        data_t = loadmat(path_t)
        data_t = list(data_t.values())[-1]
        ch_idx = 1 + var_idx * time_steps + t
        combined_data_GT[ch_idx, :, : ] = data_t

combined_data_GT = torch.tensor(combined_data_GT, dtype=torch.float64, device=device)

# S_c_GT = combined_data_GT[0].unsqueeze(0).expand(11, -1, -1)
# u_u_GT = torch.stack([combined_data_GT[i] for i in [1] + list(range(4, 14))], dim=0)  # [11, H, W]
# u_v_GT = torch.stack([combined_data_GT[i] for i in [2] + list(range(14, 24))], dim=0)
# c_flow_GT = torch.stack([combined_data_GT[i] for i in [3] + list(range(24, 34))], dim=0)


S_c_GT = combined_data_GT[0].unsqueeze(0).expand(11, -1, -1)
u_u_GT = combined_data_GT[1:12]
u_v_GT = combined_data_GT[12:23]
c_flow_GT = combined_data_GT[23:34]

# 计算每个变量的最小值和最大值
vmin_S_c, vmax_S_c = S_c_GT.min(), S_c_GT.max()
vmin_u_u, vmax_u_u = u_u_GT.min(), u_u_GT.max()
vmin_u_v, vmax_u_v = u_v_GT.min(), u_v_GT.max()
vmin_c_flow, vmax_c_flow = c_flow_GT.min(), c_flow_GT.max()

# 创建 2x6 的子图
fig, axes = plt.subplots(2, 4, figsize=(18, 8))

# 显示列表
variables = {
    'S_c': (S_c, S_c_GT, vmin_S_c, vmax_S_c),
    'u_u': (u_u, u_u_GT, vmin_u_u, vmax_u_u),
    'u_v': (u_v, u_v_GT, vmin_u_v, vmax_u_v),
    'c_flow': (c_flow, c_flow_GT, vmin_c_flow, vmax_c_flow)
}


# 绘制图像
for col, (name, (data, gt_data, vmin, vmax)) in enumerate(variables.items()):
    # 绘制预测结果

    im_pred = axes[0, col].imshow(data[2,:,:], cmap='turbo', vmin=vmin, vmax=vmax)
    axes[0, col].set_title(f'{name}')
    axes[0, col].axis('off')
    plt.colorbar(im_pred, ax=axes[0, col])

    # 绘制真实结果
    im_gt = axes[1, col].imshow(gt_data[2,:,:].cpu().numpy(), cmap='turbo', vmin=vmin, vmax=vmax)
    axes[1, col].set_title(f'{name}_GT')
    axes[1, col].axis('off')
    plt.colorbar(im_gt, ax=axes[1, col])

# 调整子图间距
plt.tight_layout()

# 显示图像
plt.show()

# 保存图像
fig.suptitle('Elder Results', y=1.02)
plt.savefig('output_Elder.png')  # 保存为 PNG 文件

# 转换为 PyTorch 张量
S_c = torch.tensor(S_c, dtype=torch.float64, device=device)
S_c_GT = torch.tensor(S_c_GT, dtype=torch.float64, device=device)
u_u= torch.tensor(u_u, dtype=torch.float64, device=device)
u_u_GT = torch.tensor(u_u_GT, dtype=torch.float64, device=device)
u_v = torch.tensor(u_v, dtype=torch.float64, device=device)
u_v_GT = torch.tensor(u_v_GT, dtype=torch.float64, device=device)
c_flow = torch.tensor(c_flow, dtype=torch.float64, device=device)
c_flow_GT = torch.tensor(c_flow_GT, dtype=torch.float64, device=device)

# 计算相对误差
def calculate_relative_error(pred, gt):
    return torch.norm(pred - gt, 2) / torch.norm(gt, 2)

relative_error_S_c = calculate_relative_error(S_c[0,:,:], S_c_GT[0,:,:])
relative_error_u_u = calculate_relative_error(u_u[2,:,:], u_u_GT[2,:,:])
relative_error_u_v = calculate_relative_error(u_v[2,:,:], u_v_GT[2,:,:])
relative_error_c_flow = calculate_relative_error(c_flow[2,:,:], c_flow_GT[2,:,:])

# 打印相对误差
print(f'Relative error of S_c: {relative_error_S_c.item():.6e}')
print(f'Relative error of u_u: {relative_error_u_u.item():.6e}')
print(f'Relative error of u_v: {relative_error_u_v.item():.6e}')
print(f'Relative error of c_flow: {relative_error_c_flow.item():.6e}')