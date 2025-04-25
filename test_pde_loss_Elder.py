import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import torch.nn.functional as F
from torch_utils import distributed as dist
import scipy.io
import os
import scipy.io as sio
import pandas as pd
from scipy.io import loadmat
import numpy as np
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt



def random_index(k, grid_size, seed=0, device=torch.device('cuda')):
    '''randomly select k indices from a [grid_size, grid_size] grid.'''
    np.random.seed(seed)
    indices = np.random.choice(grid_size**2, k, replace=False)
    indices_2d = np.unravel_index(indices, (grid_size, grid_size))
    indices_list = list(zip(indices_2d[0], indices_2d[1]))
    mask = torch.zeros((grid_size, grid_size), dtype=torch.float32).to(device)
    for i in indices_list:
        mask[i] = 1
    return mask


def get_Elder_loss(S_c, u_u, u_v, c_flow, S_c_GT, u_u_GT, u_v_GT, c_flow_GT, S_c_mask, u_u_mask, u_v_mask, c_flow_mask, device=torch.device('cuda')):
    """Return the loss of the Elder equation and the observation loss."""

    rho_0 = 1000
    beta = 200
    rho = rho_0+beta*c_flow  # [T, H, W]
    T, H, W = rho.shape

    delta_x = (300/128) # 1m
    delta_y = (150/128) # 1m
    delta_t = 2 * 365 * 24 * 60 *60 # 2 a

    # 空间导数核 (for conv2d)
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * delta_x)
    deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * delta_y)

    # 时间导数核 (for conv1d)
    deriv_t = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 3) / (2 * delta_t)

    # Darcy
    # 时间导数 d(rho)/dt
    rho_t = rho.permute(1, 2, 0).reshape(-1, 1, T)     # [H*W, 1, T]
    d_rho_dt = F.conv1d(rho_t, deriv_t, padding=1)     # [H*W, 1, T]
    d_rho_dt = d_rho_dt.squeeze(1).reshape(H, W, T).permute(2, 0, 1)  # [T, H, W]

    rho_u = rho * u_u
    rho_u_2d = rho_u.unsqueeze(1)  # [T, 1, H, W]
    d_rho_u_dx = F.conv2d(rho_u_2d, deriv_x, padding=(0, 1)).squeeze(1)  # [T, H, W]

    rho_v = rho * u_v
    rho_v_2d = rho_v.permute(0, 1, 2).unsqueeze(1)
    d_rho_v_dy = F.conv2d(rho_v_2d, deriv_y, padding=(1, 0)).squeeze(1)

    result_Darcy = 0.1 * d_rho_dt + d_rho_u_dx + d_rho_v_dy

    # TDS
    c_t = c_flow.permute(1, 2, 0).reshape(-1, 1, T)
    dc_dt = F.conv1d(c_t, deriv_t, padding=1).squeeze(1).reshape(H, W, T).permute(2, 0, 1)

    c_2d = c_flow.unsqueeze(1)  # [T, 1, H, W]
    dc_dx = F.conv2d(c_2d, deriv_x, padding=(0, 1)).squeeze(1)
    dc_dy = F.conv2d(c_2d, deriv_y, padding=(1, 0)).squeeze(1)

    laplace_c = F.conv2d(dc_dx.unsqueeze(1), deriv_x, padding=(0, 1)).squeeze(1) + F.conv2d(dc_dy.unsqueeze(1), deriv_y, padding=(1, 0)).squeeze(1)

    result_TDS = 0.1 * dc_dt + u_u * dc_dx + u_v * dc_dy - 0.1 * 3.56e-6 * laplace_c - S_c

    scipy.io.savemat('result_Darcy.mat', {'result_Darcy': result_Darcy.cpu().detach().numpy()})
    scipy.io.savemat('result_TDS.mat', {'result_TDS': result_TDS.cpu().detach().numpy()})

    pde_loss_Darcy = result_Darcy
    pde_loss_TDS = result_TDS

    pde_loss_Darcy = pde_loss_Darcy.squeeze()
    pde_loss_TDS = pde_loss_TDS.squeeze()

    # pde_loss_Darcy = pde_loss_Darcy/1
    # pde_loss_TDS = pde_loss_TDS/1


    observation_loss_S_c = (S_c - S_c_GT).squeeze()
    observation_loss_S_c = observation_loss_S_c * S_c_mask  
    observation_loss_u_u = (u_u - u_u_GT).squeeze()
    observation_loss_u_u = observation_loss_u_u * u_u_mask
    observation_loss_u_v = (u_v - u_v_GT).squeeze()
    observation_loss_u_v = observation_loss_u_v * u_v_mask  
    observation_loss_c_flow = (c_flow - c_flow_GT).squeeze()
    observation_loss_c_flow = observation_loss_c_flow * c_flow_mask  


    return pde_loss_Darcy, pde_loss_TDS, observation_loss_S_c, observation_loss_u_u, observation_loss_u_v, observation_loss_c_flow


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datapath = '/data/yangchangfan/DiffusionPDE/data/testing/Elder'
offset = 1001
time_steps = 10
C, H, W = 34, 128, 128


data_test_path = "/data/yangchangfan/DiffusionPDE/data/testing/Elder/"


combined_data_GT = np.zeros(( C, H, W), dtype=np.float64)

# ---------- 读取 S_c ----------
path_Sc = os.path.join(data_test_path, 'S_c', str(offset), '0.mat')
Sc_data = loadmat(path_Sc)
Sc = list(Sc_data.values())[-1]
combined_data_GT[0, :, :] = Sc

# ---------- 读取初始场 + 时域场 ----------
var_names = ['u_u', 'u_v', 'c_flow']
for var_idx, var in enumerate(var_names):
    # 初始场（通道 1 ~ 3）
    path_0 = os.path.join(data_test_path, var, str(offset), '0.mat')
    data0 = loadmat(path_0)
    data0 = list(data0.values())[-1]
    combined_data_GT[1 + var_idx, :, :] = data0

    # 时域场 t=1~10（通道 4 ~ 33）
    for t in range(1, time_steps + 1):
        path_t = os.path.join(data_test_path, var, str(offset), f'{t}.mat')
        data_t = loadmat(path_t)
        data_t = list(data_t.values())[-1]
        ch_idx = 4 + var_idx * time_steps + (t - 1)
        combined_data_GT[ch_idx, :, : ] = data_t

combined_data_GT = torch.tensor(combined_data_GT, dtype=torch.float64, device=device)


known_index_S_c = random_index(500, 128, seed=3)
known_index_u_u = random_index(500, 128, seed=2)
known_index_u_v = random_index(500, 128, seed=1)
known_index_c_flow = random_index(500, 128, seed=0)


S_c_N = combined_data_GT[0].unsqueeze(0).expand(11, -1, -1)
u_u_N = torch.stack([combined_data_GT[i] for i in [1] + list(range(4, 14))], dim=0)  # [11, H, W]
u_v_N = torch.stack([combined_data_GT[i] for i in [2] + list(range(14, 24))], dim=0)
c_flow_N = torch.stack([combined_data_GT[i] for i in [3] + list(range(24, 34))], dim=0)

S_c_GT = S_c_N
u_u_GT = u_u_N
u_v_GT = u_v_N
c_flow_GT = c_flow_N

pde_loss_Darcy, pde_loss_TDS, observation_loss_S_c, observation_loss_u_u, observation_loss_u_v, observation_loss_c_flow = get_Elder_loss(S_c_N, u_u_N, u_v_N, c_flow_N, S_c_GT, u_u_GT, u_v_GT, c_flow_GT, known_index_S_c, known_index_u_u, known_index_u_v, known_index_c_flow, device=device)

L_pde_Darcy = torch.norm(pde_loss_Darcy, 2)/(128*128)
L_pde_TDS = torch.norm(pde_loss_TDS, 2)/(128*128)

L_obs_S_c = torch.norm(observation_loss_S_c, 2)/500
L_obs_u_u = torch.norm(observation_loss_u_u, 2)/500
L_obs_u_v = torch.norm(observation_loss_u_v, 2)/500
L_obs_c_flow = torch.norm(observation_loss_c_flow, 2)/500


print(L_pde_Darcy)
print(L_pde_TDS)

print(L_obs_S_c)
print(L_obs_u_u)
print(L_obs_u_v)
print(L_obs_c_flow)

# scipy.io.savemat('L_pde_NS.mat', {'L_pde_NS': L_pde_NS.cpu().detach().numpy()})
# scipy.io.savemat('L_pde_heat.mat', {'L_pde_heat': L_pde_heat.cpu().detach().numpy()})


pde_loss_Darcy_np = pde_loss_Darcy.cpu().numpy()
pde_loss_TDS_np = pde_loss_TDS.cpu().numpy()


scipy.io.savemat('pde_loss_Darcy.mat', {'pde_loss_Darcy_x': pde_loss_Darcy.cpu().detach().numpy()})
scipy.io.savemat('pde_loss_TDS.mat', {'pde_loss_TDS': pde_loss_TDS.cpu().detach().numpy()})



plt.figure(figsize=(8, 6))
plt.imshow(pde_loss_Darcy_np[5,:,:], cmap='viridis')  # 使用 viridis 色彩映射
plt.colorbar()  # 添加颜色条
plt.title('Elder Loss for Darcy')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('Elder_loss_Darcy.png')  # 保存为 PNG 文件

plt.figure(figsize=(8, 6))
plt.imshow(pde_loss_TDS_np[5,:,:], cmap='viridis')  # 使用 viridis 色彩映射
plt.colorbar()  # 添加颜色条
plt.title('Elder Loss for TDS')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('Elder_loss_TDS.png')  # 保存为 PNG 文件

