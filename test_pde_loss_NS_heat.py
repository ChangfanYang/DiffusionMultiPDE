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

def identify_mater(circle_params, device=torch.device('cuda')):

    mater_iden = torch.zeros(128,128, device=device)
    circle_params = circle_params.squeeze(0)
    cx, cy, r = map(float, circle_params) 
    coords = (torch.arange(128, device=device) - 63.5) * 0.001
    xx, yy = torch.meshgrid(coords, coords, indexing='ij')

    mater_iden = torch.where((xx-cx)**2 + (yy-cy)**2 <= r**2, 1, -1)
    # in 1, out -1
    return mater_iden


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


def generate_separa_PDE_mater(mater_iden, device=torch.device('cuda')):

    rho_air = 1.24246
    rho_copper = 8960
    Crho_air = 1005.10779
    Crho_copper = 385
    kappa_air = 0.02505
    kappa_copper = 400

    rho = torch.where(mater_iden > 1e-5, rho_copper, rho_air)
    Crho = torch.where(mater_iden > 1e-5, Crho_copper, Crho_air)
    kappa = torch.where(mater_iden > 1e-5, kappa_copper, kappa_air)

    rho = rho.t()
    Crho = Crho.t()
    kappa = kappa.t()

    # scipy.io.savemat('pho.mat', {'pho': pho.cpu().detach().numpy()})
    # scipy.io.savemat('Cpho.mat', {'Cpho': Cpho.cpu().detach().numpy()})
    # scipy.io.savemat('kappa.mat', {'kappa': kappa.cpu().detach().numpy()})

    return rho, Crho, kappa


def get_NS_heat_loss(Q_heat, u_u, u_v, T, Q_heat_GT, u_u_GT, u_v_GT, T_GT, Q_heat_mask, u_u_mask, u_v_mask, T_mask, mater_iden, device=torch.device('cuda')):
    """Return the loss of the NS_heat equation and the observation loss."""

    rho, Crho, kappa = generate_separa_PDE_mater(mater_iden)

    delta_x = 0.128/128 # 1mm
    delta_y = 0.128/128 # 1mm
    
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * delta_x)
    deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * delta_y)

    # Continuity_NS
    grad_x_next_x_NS = F.conv2d(u_u, deriv_x, padding=(0, 1))
    grad_x_next_y_NS = F.conv2d(u_v, deriv_y, padding=(1, 0))
    result_NS = grad_x_next_x_NS + grad_x_next_y_NS

    # T_filed
    grad_x_next_x_T = F.conv2d(T, deriv_x, padding=(0, 1))
    grad_x_next_y_T = F.conv2d(T, deriv_y, padding=(1, 0))
    Laplac_T = F.conv2d(grad_x_next_x_T, deriv_x, padding=(0, 1)) + F.conv2d(grad_x_next_y_T, deriv_y, padding=(1, 0))

    result_heat = rho * Crho * (u_u * grad_x_next_x_T + u_v * grad_x_next_y_T) - kappa * Laplac_T - Q_heat
    # result_heat = rho * Crho * (u_u * grad_x_next_x_T + u_v * grad_x_next_y_T) - kappa * Laplac_T
    

    pde_loss_NS = result_NS
    pde_loss_heat = result_heat

    pde_loss_NS = pde_loss_NS.squeeze()
    pde_loss_heat = pde_loss_heat.squeeze()
    
    pde_loss_heat = pde_loss_heat/1000000
    pde_loss_NS = pde_loss_NS/1000

    scipy.io.savemat('grad_x_next_x_T.mat', {'grad_x_next_x_T': grad_x_next_x_T.cpu().detach().numpy()})
    scipy.io.savemat('grad_x_next_y_T.mat', {'grad_x_next_y_T': grad_x_next_y_T.cpu().detach().numpy()})
    scipy.io.savemat('T.mat', {'T': T.cpu().detach().numpy()})
    scipy.io.savemat('Laplac_T.mat', {'Laplac_T': Laplac_T.cpu().detach().numpy()})
    scipy.io.savemat('result_heat.mat', {'result_heat': result_heat.cpu().detach().numpy()})
    scipy.io.savemat('test_Q_heat.mat', {'Q_heat': Q_heat.cpu().detach().numpy()})
    scipy.io.savemat('test_u_u.mat', {'u_u': u_u.cpu().detach().numpy()})
    scipy.io.savemat('test_u_v.mat', {'u_u': u_v.cpu().detach().numpy()})


    observation_loss_Q_heat = (Q_heat - Q_heat_GT).squeeze()
    observation_loss_Q_heat = observation_loss_Q_heat * Q_heat_mask  
    observation_loss_u_u = (u_u - u_u_GT).squeeze()
    observation_loss_u_u = observation_loss_u_u * u_u_mask  
    observation_loss_u_v = (u_v - u_v_GT).squeeze()
    observation_loss_u_v = observation_loss_u_v * u_v_mask  
    observation_loss_T = (T - T_GT).squeeze()
    observation_loss_T = observation_loss_T * T_mask

    return pde_loss_NS, pde_loss_heat, observation_loss_Q_heat, observation_loss_u_u, observation_loss_u_v, observation_loss_T


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datapath = '/data/yangchangfan/DiffusionPDE/data/testing/NS_heat'
offset = 10001

Q_heat_GT_path = os.path.join(datapath, "Q_heat", f"{offset}.mat")
Q_heat_GT = sio.loadmat(Q_heat_GT_path)['export_Q_heat']
Q_heat_GT = torch.tensor(Q_heat_GT, dtype=torch.float64, device=device)

u_u_GT_path = os.path.join(datapath, "u_u", f"{offset}.mat")
u_u_GT = sio.loadmat(u_u_GT_path)['export_u_u']
u_u_GT = torch.tensor(u_u_GT, device=device)
    
u_v_GT_path = os.path.join(datapath, "u_v", f"{offset}.mat")
u_v_GT = sio.loadmat(u_v_GT_path)['export_u_v']
u_v_GT = torch.tensor(u_v_GT, dtype=torch.float64, device=device)

T_GT_path = os.path.join(datapath, "T", f"{offset}.mat")
T_GT = sio.loadmat(T_GT_path)['export_T']
T_GT = torch.tensor(T_GT, dtype=torch.float64, device=device)
    
circle_GT_path = os.path.join(datapath, "circlecsv", f"{offset}.csv")
circle_GT = pd.read_csv(circle_GT_path, header=None)
circle_GT = torch.tensor(circle_GT.values, dtype=torch.float64)

known_index_Q_heat = random_index(500, 128, seed=3)
known_index_u_u = random_index(500, 128, seed=2)
known_index_u_v = random_index(500, 128, seed=1)
known_index_T = random_index(500, 128, seed=0)

circle_iden = identify_mater(circle_GT)

Q_heat_N = Q_heat_GT.unsqueeze(0).unsqueeze(0)
u_u_N = u_u_GT.unsqueeze(0).unsqueeze(0)
u_v_N = u_v_GT.unsqueeze(0).unsqueeze(0)
T_N = T_GT.unsqueeze(0).unsqueeze(0)  

pde_loss_NS, pde_loss_heat, observation_loss_Q_heat, observation_loss_u_u, observation_loss_u_v, observation_loss_T = get_NS_heat_loss(Q_heat_N, u_u_N, u_v_N, T_N, Q_heat_GT, u_u_GT, u_v_GT, T_GT, known_index_Q_heat, known_index_u_u, known_index_u_v, known_index_T, circle_iden, device=device)
        

L_pde_NS = torch.norm(pde_loss_NS, 2)/(128*128)
L_pde_heat = torch.norm(pde_loss_heat, 2)/(128*128)

L_obs_Q_heat = torch.norm(observation_loss_Q_heat, 2)/500
L_obs_u_u = torch.norm(observation_loss_u_u, 2)/500
L_obs_u_v = torch.norm(observation_loss_u_v, 2)/500
L_obs_T = torch.norm(observation_loss_T, 2)/500

print(L_pde_NS)
print(L_pde_heat)


# scipy.io.savemat('L_pde_NS.mat', {'L_pde_NS': L_pde_NS.cpu().detach().numpy()})
# scipy.io.savemat('L_pde_heat.mat', {'L_pde_heat': L_pde_heat.cpu().detach().numpy()})


pde_loss_NS = pde_loss_NS.cuda()
pde_loss_heat = pde_loss_heat.cuda()

pde_loss_NS_np = pde_loss_NS.cpu().numpy()
pde_loss_heat_np = pde_loss_heat.cpu().numpy()

plt.figure(figsize=(8, 6))
plt.imshow(pde_loss_NS_np, cmap='viridis',vmin=-0.0001,vmax=0.0001)  # 使用 viridis 色彩映射
plt.colorbar()  # 添加颜色条
plt.title('NS_heat PDE Loss-NS')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('NS_heat_loss_NS.png')  # 保存为 PNG 文件


# 绘制 pde_loss_heat 的图像
plt.figure(figsize=(8, 6))
plt.imshow(pde_loss_heat_np, cmap='viridis',vmin=-0.001,vmax=0.001)  # 使用相同的色彩映射
plt.colorbar()
plt.title('NS_heat PDE heat')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('NS_heat_loss_heat.png')  # 保存为 PNG 文件

