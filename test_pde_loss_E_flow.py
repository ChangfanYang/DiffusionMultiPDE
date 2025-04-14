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


def get_E_flow_loss(kappa, ec_V, u_flow, v_flow, kappa_GT, ec_V_GT, u_flow_GT, v_flow_GT, kappa_mask, ec_V_mask, u_flow_mask, v_flow_mask, device=torch.device('cuda')):
    """Return the loss of the E_flow equation and the observation loss."""

    delta_x = 1e-3 # 1mm
    delta_y = 1e-3 # 1mm
    
    deriv_x = torch.tensor([[1, 0, -1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * delta_x)
    deriv_y = torch.tensor([[1], [0], [-1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * delta_y)

    # Continuity_NS
    grad_x_next_x_NS = F.conv2d(u_flow, deriv_x, padding=(0, 1))
    grad_x_next_y_NS = F.conv2d(v_flow, deriv_y, padding=(1, 0))
    result_NS = grad_x_next_x_NS + grad_x_next_y_NS

    # Continuity_J
    grad_x_next_x_V = F.conv2d(ec_V, deriv_x, padding=(0, 1))
    grad_x_next_y_V = F.conv2d(ec_V, deriv_y, padding=(1, 0))

    grad_x_next_x_J = F.conv2d(kappa*grad_x_next_x_V, deriv_x, padding=(0, 1))
    grad_x_next_y_J = F.conv2d(kappa*grad_x_next_y_V, deriv_y, padding=(1, 0))

    result_J = grad_x_next_x_J + grad_x_next_y_J
    
    pde_loss_NS = result_NS
    pde_loss_J = result_J

    pde_loss_NS = pde_loss_NS.squeeze()
    pde_loss_J = pde_loss_J.squeeze()
    
    pde_loss_NS = pde_loss_NS/1000
    pde_loss_J = pde_loss_J/1000000


    # scipy.io.savemat('test_rho.mat', {'rho': rho.cpu().detach().numpy()})
    # scipy.io.savemat('test_Crho.mat', {'Crho': Crho.cpu().detach().numpy()})
    # scipy.io.savemat('test_kappa.mat', {'kappa': kappa.cpu().detach().numpy()})
    # scipy.io.savemat('test_Laplac_T.mat', {'Laplac_T': Laplac_T.cpu().detach().numpy()})
    # scipy.io.savemat('test_Q_heat.mat', {'Q_heat': Q_heat.cpu().detach().numpy()})
    # scipy.io.savemat('test_u_flow.mat', {'u_flow': u_flow.cpu().detach().numpy()})
    # scipy.io.savemat('test_v_flow.mat', {'v_flow': v_flow.cpu().detach().numpy()})


    observation_loss_kappa = (kappa - kappa_GT).squeeze()
    observation_loss_kappa = observation_loss_kappa * kappa_mask  
    observation_loss_ec_V = (ec_V - ec_V_GT).squeeze()
    observation_loss_ec_V = observation_loss_ec_V * ec_V_mask
    observation_loss_u_flow = (u_flow - u_flow_GT).squeeze()
    observation_loss_u_flow = observation_loss_u_flow * u_flow_mask  
    observation_loss_v_flow = (v_flow - v_flow_GT).squeeze()
    observation_loss_v_flow = observation_loss_v_flow * v_flow_mask  

    return pde_loss_NS, pde_loss_J, observation_loss_kappa, observation_loss_ec_V, observation_loss_u_flow, observation_loss_v_flow


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datapath = '/data/yangchangfan/DiffusionPDE/data/testing/E_flow'
offset = 10001

kappa_GT_path = os.path.join(datapath, "kappa", f"{offset}.mat")
kappa_GT = sio.loadmat(kappa_GT_path)['export_kappa']
kappa_GT = torch.tensor(kappa_GT, dtype=torch.float64, device=device)

ec_V_GT_path = os.path.join(datapath, "ec_V", f"{offset}.mat")
ec_V_GT = sio.loadmat(ec_V_GT_path)['export_ec_V']
ec_V_GT = torch.tensor(ec_V_GT, dtype=torch.float64, device=device)

u_flow_GT_path = os.path.join(datapath, "u_flow", f"{offset}.mat")
u_flow_GT = sio.loadmat(u_flow_GT_path)['export_u_flow']
u_flow_GT = torch.tensor(u_flow_GT, device=device)
    
v_flow_GT_path = os.path.join(datapath, "v_flow", f"{offset}.mat")
v_flow_GT = sio.loadmat(v_flow_GT_path)['export_v_flow']
v_flow_GT = torch.tensor(v_flow_GT, dtype=torch.float64, device=device)


known_index_kappa = random_index(500, 128, seed=3)
known_index_ec_V = random_index(500, 128, seed=2)
known_index_u_flow = random_index(500, 128, seed=1)
known_index_v_flow = random_index(500, 128, seed=0)



kappa_N = kappa_GT.unsqueeze(0).unsqueeze(0)
ec_V_N = ec_V_GT.unsqueeze(0).unsqueeze(0)
u_flow_N = u_flow_GT.unsqueeze(0).unsqueeze(0)
v_flow_N = v_flow_GT.unsqueeze(0).unsqueeze(0)


pde_loss_NS, pde_loss_J, observation_loss_kappa, observation_loss_ec_V, observation_loss_u_flow, observation_loss_v_flow = get_E_flow_loss(kappa_N, ec_V_N, u_flow_N, v_flow_N, kappa_GT, ec_V_GT, u_flow_GT, v_flow_GT, known_index_kappa, known_index_ec_V, known_index_u_flow, known_index_v_flow, device=device)
             

L_pde_NS = torch.norm(pde_loss_NS, 2)/(128*128)
L_pde_J = torch.norm(pde_loss_J, 2)/(128*128)

L_obs_kappa = torch.norm(observation_loss_kappa, 2)/500
L_obs_ec_V = torch.norm(observation_loss_ec_V, 2)/500
L_obs_u_flow = torch.norm(observation_loss_u_flow, 2)/500
L_obs_v_flow = torch.norm(observation_loss_v_flow, 2)/500


print(L_pde_NS)
print(L_pde_J)


# scipy.io.savemat('L_pde_NS.mat', {'L_pde_NS': L_pde_NS.cpu().detach().numpy()})
# scipy.io.savemat('L_pde_heat.mat', {'L_pde_heat': L_pde_heat.cpu().detach().numpy()})


pde_loss_NS = pde_loss_NS.cuda()
pde_loss_J = pde_loss_J.cuda()

pde_loss_NS_np = pde_loss_NS.cpu().numpy()
pde_loss_J_np = pde_loss_J.cpu().numpy()

plt.figure(figsize=(8, 6))
plt.imshow(pde_loss_NS_np, cmap='viridis')  # 使用 viridis 色彩映射
plt.colorbar()  # 添加颜色条
plt.title('E_flow Loss for NS')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('E_flow_loss_NS.png')  # 保存为 PNG 文件


# 绘制 pde_loss_J 的图像
plt.figure(figsize=(8, 6))
plt.imshow(pde_loss_J_np, cmap='viridis')  # 使用相同的色彩映射
plt.colorbar()
plt.title('PDE Loss for J')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('E_flow_loss_heat.png')  # 保存为 PNG 文件

