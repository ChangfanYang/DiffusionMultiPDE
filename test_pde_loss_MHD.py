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


def get_MHD_loss(Br, Jx, Jy, Jz, u_u, u_v, Br_GT, Jx_GT, Jy_GT, Jz_GT, u_u_GT, u_v_GT, Br_mask, Jx_mask, Jy_mask, Jz_mask, u_u_mask, u_v_mask, device=torch.device('cuda')):
    """Return the loss of the MHD equation and the observation loss."""

    delta_x = 1e-2 # 1cm
    delta_y = 1e-2 # 1cm
    
    deriv_x = torch.tensor([[1, 0, -1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * delta_x)
    deriv_y = torch.tensor([[1], [0], [-1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * delta_y)

    # Continuity_NS
    grad_x_next_x_NS = F.conv2d(u_u, deriv_x, padding=(0, 1))
    grad_x_next_y_NS = F.conv2d(u_v, deriv_y, padding=(1, 0))
    result_NS = grad_x_next_x_NS + grad_x_next_y_NS

    # Continuity_J
    grad_x_next_x_J = F.conv2d(Jx, deriv_x, padding=(0, 1))
    grad_x_next_y_J = F.conv2d(Jy, deriv_y, padding=(1, 0))
    result_J = grad_x_next_x_J + grad_x_next_y_J
    
    pde_loss_NS = result_NS
    pde_loss_J = result_J

    pde_loss_NS = pde_loss_NS.squeeze()
    pde_loss_J = pde_loss_J.squeeze()
    
    pde_loss_NS = pde_loss_NS/100
    pde_loss_J = pde_loss_J/100


    # scipy.io.savemat('test_rho.mat', {'rho': rho.cpu().detach().numpy()})
    # scipy.io.savemat('test_Crho.mat', {'Crho': Crho.cpu().detach().numpy()})
    # scipy.io.savemat('test_kappa.mat', {'kappa': kappa.cpu().detach().numpy()})
    # scipy.io.savemat('test_Laplac_T.mat', {'Laplac_T': Laplac_T.cpu().detach().numpy()})
    # scipy.io.savemat('test_Q_heat.mat', {'Q_heat': Q_heat.cpu().detach().numpy()})
    # scipy.io.savemat('test_u_u.mat', {'u_u': u_u.cpu().detach().numpy()})
    # scipy.io.savemat('test_u_v.mat', {'u_u': u_v.cpu().detach().numpy()})


    observation_loss_Br = (Br - Br_GT).squeeze()
    observation_loss_Br = observation_loss_Br * Br_mask  
    observation_loss_Jx = (Jx - Jx_GT).squeeze()
    observation_loss_Jx = observation_loss_Jx * Jx_mask
    observation_loss_Jy = (Jy - Jy_GT).squeeze()
    observation_loss_Jy = observation_loss_Jy * Jy_mask
    observation_loss_Jz = (Jz - Jz_GT).squeeze()
    observation_loss_Jz = observation_loss_Jz * Jz_mask
    observation_loss_u_u = (u_u - u_u_GT).squeeze()
    observation_loss_u_u = observation_loss_u_u * u_u_mask  
    observation_loss_u_v = (u_v - u_v_GT).squeeze()
    observation_loss_u_v = observation_loss_u_v * u_v_mask  

    return pde_loss_NS, pde_loss_J, observation_loss_Br, observation_loss_Jx, observation_loss_Jy, observation_loss_Jz, observation_loss_u_u, observation_loss_u_v

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datapath = '/data/yangchangfan/DiffusionPDE/data/testing/MHD'
offset = 10001

Br_GT_path = os.path.join(datapath, "Br", f"{offset}.mat")
Br_GT = sio.loadmat(Br_GT_path)['export_Br']
Br_GT = torch.tensor(Br_GT, dtype=torch.float64, device=device)

Jx_GT_path = os.path.join(datapath, "Jx", f"{offset}.mat")
Jx_GT = sio.loadmat(Jx_GT_path)['export_Jx']
Jx_GT = torch.tensor(Jx_GT, dtype=torch.float64, device=device)

Jy_GT_path = os.path.join(datapath, "Jy", f"{offset}.mat")
Jy_GT = sio.loadmat(Jy_GT_path)['export_Jy']
Jy_GT = torch.tensor(Jy_GT, dtype=torch.float64, device=device)

Jz_GT_path = os.path.join(datapath, "Jz", f"{offset}.mat")
Jz_GT = sio.loadmat(Jz_GT_path)['export_Jz']
Jz_GT = torch.tensor(Jz_GT, dtype=torch.float64, device=device)

u_u_GT_path = os.path.join(datapath, "u_u", f"{offset}.mat")
u_u_GT = sio.loadmat(u_u_GT_path)['export_u']
u_u_GT = torch.tensor(u_u_GT, device=device)
    
u_v_GT_path = os.path.join(datapath, "u_v", f"{offset}.mat")
u_v_GT = sio.loadmat(u_v_GT_path)['export_v']
u_v_GT = torch.tensor(u_v_GT, dtype=torch.float64, device=device)


known_index_Br = random_index(500, 128, seed=5)
known_index_Jx = random_index(500, 128, seed=4)
known_index_Jy = random_index(500, 128, seed=3)
known_index_Jz = random_index(500, 128, seed=2)
known_index_u_u = random_index(500, 128, seed=1)
known_index_u_v = random_index(500, 128, seed=0)



Br_N = Br_GT.unsqueeze(0).unsqueeze(0)
Jx_N = Jx_GT.unsqueeze(0).unsqueeze(0)
Jy_N = Jy_GT.unsqueeze(0).unsqueeze(0)
Jz_N = Jz_GT.unsqueeze(0).unsqueeze(0)
u_u_N = u_u_GT.unsqueeze(0).unsqueeze(0)
u_v_N = u_v_GT.unsqueeze(0).unsqueeze(0)


pde_loss_NS, pde_loss_J, observation_loss_Br, observation_loss_Jx, observation_loss_Jy, observation_loss_Jz, observation_loss_u_u, observation_loss_u_v = get_MHD_loss(Br_N, Jx_N, Jy_N, Jz_N, u_u_N, u_v_N, Br_GT, Jx_GT, Jy_GT, Jz_GT, u_u_GT, u_v_GT, known_index_Br, known_index_Jx, known_index_Jy, known_index_Jz, known_index_u_u, known_index_u_v, device=device)
        

L_pde_NS = torch.norm(pde_loss_NS, 2)/(128*128)
L_pde_J = torch.norm(pde_loss_J, 2)/(128*128)

L_obs_Br = torch.norm(observation_loss_Br, 2)/500
L_obs_Jx = torch.norm(observation_loss_Jx, 2)/500
L_obs_Jy = torch.norm(observation_loss_Jy, 2)/500
L_obs_Jz = torch.norm(observation_loss_Jz, 2)/500
L_obs_u_u = torch.norm(observation_loss_u_u, 2)/500
L_obs_u_v = torch.norm(observation_loss_u_v, 2)/500


print(L_pde_NS)
print(L_pde_J)


# scipy.io.savemat('L_pde_NS.mat', {'L_pde_NS': L_pde_NS.cpu().detach().numpy()})
# scipy.io.savemat('L_pde_heat.mat', {'L_pde_heat': L_pde_heat.cpu().detach().numpy()})


pde_loss_NS = pde_loss_NS.cuda()
pde_loss_J = pde_loss_J.cuda()

pde_loss_NS_np = pde_loss_NS.cpu().numpy()
pde_loss_J_np = pde_loss_J.cpu().numpy()


vmin_pde_loss_NS_np, vmax_pde_loss_NS_np = pde_loss_NS_np.min(), pde_loss_NS_np.max()
vmin_pde_loss_J_np, vmax_pde_loss_J_np = pde_loss_J_np.min(), pde_loss_J_np.max()



plt.figure(figsize=(8, 6))
plt.imshow(pde_loss_NS_np, cmap='viridis', vmin=vmin_pde_loss_NS_np, vmax=vmax_pde_loss_NS_np)  # 使用 viridis 色彩映射
plt.colorbar()  # 添加颜色条
plt.title('MHD Loss for NS')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('MHD_loss_NS.png')  # 保存为 PNG 文件


# 绘制 pde_loss_J 的图像
plt.figure(figsize=(8, 6))
plt.imshow(pde_loss_J_np, cmap='viridis', vmin=vmin_pde_loss_J_np/100, vmax=vmax_pde_loss_J_np/100) # 使用相同的色彩映射
plt.colorbar()
plt.title('PDE Loss for J')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('MHD_loss_heat.png')  # 保存为 PNG 文件

