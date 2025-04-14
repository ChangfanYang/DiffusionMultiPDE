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


def get_VA_loss(rho_water, p_t, Sxx, Sxy, Syy, x_u, x_v, rho_water_GT, p_t_GT, Sxx_GT, Sxy_GT, Syy_GT, x_u_GT, x_v_GT, rho_water_mask, p_t_mask, Sxx_mask, Sxy_mask, Syy_mask, x_u_mask, x_v_mask, device=torch.device('cuda')):
    """Return the loss of the VA equation and the observation loss."""

    omega = torch.tensor(np.pi * 1e5, dtype=torch.float64, device=device)
    c_ac = 1.48144e3
    rho_Aluminum = 2730

    p_t_real = p_t.real.to(dtype=torch.float64)
    p_t_imag = p_t.imag.to(dtype=torch.float64)
    Sxx_real = Sxx.real.to(dtype=torch.float64)
    Sxx_imag = Sxx.imag.to(dtype=torch.float64)
    Sxy_real = Sxy.real.to(dtype=torch.float64)
    Sxy_imag = Sxy.imag.to(dtype=torch.float64)
    Syy_real = Syy.real.to(dtype=torch.float64)
    Syy_imag = Syy.imag.to(dtype=torch.float64)

    x_u_real = x_u.real.to(dtype=torch.float64)
    x_u_imag = x_u.imag.to(dtype=torch.float64)
    x_v_real = x_v.real.to(dtype=torch.float64)
    x_v_imag = x_v.imag.to(dtype=torch.float64)


    delta_x = (40/128)*1e-3 # 1mm
    delta_y = (40/128)*1e-3 # 1mm
    
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * delta_x)
    deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * delta_y)

    # Continuity_acoustic real
    grad_x_next_x_p_t_real = F.conv2d(p_t_real, deriv_x, padding=(0, 1))
    grad_x_next_y_p_t_real = F.conv2d(p_t_real, deriv_y, padding=(1, 0))
    Laplace_p_t_real = F.conv2d(grad_x_next_x_p_t_real/rho_water, deriv_x, padding=(0, 1)) + F.conv2d(grad_x_next_y_p_t_real/rho_water, deriv_y, padding=(1, 0))
    result_AC_real = Laplace_p_t_real + omega**2*p_t_real/(rho_water*c_ac**2)

    # Continuity_acoustic imag
    grad_x_next_x_p_t_imag = F.conv2d(p_t_imag, deriv_x, padding=(0, 1))
    grad_x_next_y_p_t_imag = F.conv2d(p_t_imag, deriv_y, padding=(1, 0))
    Laplace_p_t_imag = F.conv2d(grad_x_next_x_p_t_imag/rho_water, deriv_x, padding=(0, 1)) + F.conv2d(grad_x_next_y_p_t_imag/rho_water, deriv_y, padding=(1, 0))
    result_AC_imag = Laplace_p_t_imag + omega**2*p_t_imag/(rho_water*c_ac**2)


    # Continuity_structure real_x imag_x
    grad_x_next_x_Sxx_real = F.conv2d(Sxx_real, deriv_x, padding=(0, 1))
    grad_x_next_y_Sxy_real = F.conv2d(Sxy_real, deriv_y, padding=(1, 0))
    result_structure_real_x = grad_x_next_x_Sxx_real + grad_x_next_y_Sxy_real + rho_Aluminum * omega**2 * x_u_real
    
    grad_x_next_x_Sxx_imag = F.conv2d(Sxx_imag, deriv_x, padding=(0, 1))
    grad_x_next_y_Sxy_imag = F.conv2d(Sxy_imag, deriv_y, padding=(1, 0))
    result_structure_imag_x = grad_x_next_x_Sxx_imag + grad_x_next_y_Sxy_imag + rho_Aluminum * omega**2 * x_u_imag

    # Continuity_structure real_y imag_y
    grad_x_next_x_Sxy_real = F.conv2d(Sxy_real, deriv_x, padding=(0, 1))
    grad_x_next_y_Syy_real = F.conv2d(Syy_real, deriv_y, padding=(1, 0))
    result_structure_real_y = grad_x_next_x_Sxy_real + grad_x_next_y_Syy_real + rho_Aluminum * omega**2 * x_v_real

    grad_x_next_x_Sxy_imag = F.conv2d(Sxy_imag, deriv_x, padding=(0, 1))
    grad_x_next_y_Syy_imag = F.conv2d(Syy_imag, deriv_y, padding=(1, 0))
    result_structure_imag_y = grad_x_next_x_Sxy_imag + grad_x_next_y_Syy_imag + rho_Aluminum * omega**2 * x_v_imag


    scipy.io.savemat('grad_x_next_x_Sxx_real.mat', {'grad_x_next_x_Sxx_real': grad_x_next_x_Sxx_real.cpu().detach().numpy()})
    scipy.io.savemat('grad_x_next_y_Sxy_real.mat', {'grad_x_next_y_Sxy_real': grad_x_next_y_Sxy_real.cpu().detach().numpy()})
    # scipy.io.savemat('result_part_real.mat', {'result_part_real': result_part_real.cpu().detach().numpy()})
    # scipy.io.savemat('x_u_real.mat', {'x_u_real': x_u_real.cpu().detach().numpy()})
    # scipy.io.savemat('x_u_imag.mat', {'x_u_imag': x_u_imag.cpu().detach().numpy()})
    scipy.io.savemat('x_u_real.mat', {'x_u_real': x_u_real.cpu().detach().numpy()})
    scipy.io.savemat('Sxx_real.mat', {'Sxx_real': Sxx_real.cpu().detach().numpy()})

    pde_loss_AC_real = result_AC_real
    pde_loss_AC_imag = result_AC_imag

    pde_loss_structure_real_x = result_structure_real_x
    pde_loss_structure_imag_x = result_structure_imag_x

    pde_loss_structure_real_y = result_structure_real_y
    pde_loss_structure_imag_y = result_structure_imag_y


    pde_loss_AC_real = pde_loss_AC_real.squeeze()
    pde_loss_AC_imag = pde_loss_AC_imag.squeeze()

    pde_loss_structure_real_x = pde_loss_structure_real_x.squeeze()
    pde_loss_structure_imag_x = pde_loss_structure_imag_x.squeeze()
    pde_loss_structure_real_y = pde_loss_structure_real_y.squeeze()
    pde_loss_structure_imag_y = pde_loss_structure_imag_y.squeeze()
    
    pde_loss_AC_real = pde_loss_AC_real/1000000
    pde_loss_AC_imag = pde_loss_AC_imag/1000000

    pde_loss_structure_real_x = pde_loss_structure_real_x/1000
    pde_loss_structure_imag_x = pde_loss_structure_imag_x/1000
    pde_loss_structure_real_y = pde_loss_structure_real_y/1000
    pde_loss_structure_imag_y = pde_loss_structure_imag_y/1000


    # scipy.io.savemat('test_rho.mat', {'rho': rho.cpu().detach().numpy()})
    # scipy.io.savemat('test_Crho.mat', {'Crho': Crho.cpu().detach().numpy()})
    # scipy.io.savemat('test_rho_water.mat', {'rho_watera': rho_water.cpu().detach().numpy()})
    # scipy.io.savemat('test_Laplac_T.mat', {'Laplac_T': Laplac_T.cpu().detach().numpy()})
    # scipy.io.savemat('test_Q_heat.mat', {'Q_heat': Q_heat.cpu().detach().numpy()})
    # scipy.io.savemat('test_x_u.mat', {'x_u': x_u.cpu().detach().numpy()})
    # scipy.io.savemat('test_x_v.mat', {'x_v': x_v.cpu().detach().numpy()})


    observation_loss_rho_water = (rho_water - rho_water_GT).squeeze()
    observation_loss_rho_water = observation_loss_rho_water * rho_water_mask  
    observation_loss_p_t = (p_t - p_t_GT).squeeze()
    observation_loss_p_t = observation_loss_p_t * p_t_mask
    observation_loss_Sxx = (Sxx - Sxx_GT).squeeze()
    observation_loss_Sxx = observation_loss_Sxx * Sxx_mask  
    observation_loss_Sxy = (Sxy - Sxy_GT).squeeze()
    observation_loss_Sxy = observation_loss_Sxy * Sxy_mask  
    observation_loss_Syy = (Syy - Syy_GT).squeeze()
    observation_loss_Syy = observation_loss_Syy * Syy_mask  
    observation_loss_x_u = (x_u - x_u_GT).squeeze()
    observation_loss_x_u = observation_loss_x_u * x_u_mask  
    observation_loss_x_v = (x_v - x_v_GT).squeeze()
    observation_loss_x_v = observation_loss_x_v * x_v_mask  

    return pde_loss_AC_real, pde_loss_AC_imag, pde_loss_structure_real_x, pde_loss_structure_imag_x, pde_loss_structure_real_y, pde_loss_structure_imag_y, observation_loss_rho_water, observation_loss_p_t, observation_loss_Sxx, observation_loss_Sxy, observation_loss_Syy, observation_loss_x_u, observation_loss_x_v


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datapath = '/data/yangchangfan/DiffusionPDE/data/testing/VA'
offset = 10002

rho_water_GT_path = os.path.join(datapath, "rho_water", f"{offset}.mat")
rho_water_GT = sio.loadmat(rho_water_GT_path)['export_rho_water']
rho_water_GT = torch.tensor(rho_water_GT, dtype=torch.float64, device=device)

p_t_GT_path = os.path.join(datapath, "p_t", f"{offset}.mat")
p_t_GT = sio.loadmat(p_t_GT_path)['export_p_t']
p_t_GT = torch.tensor(p_t_GT, dtype=torch.complex128, device=device)


Sxx_GT_path = os.path.join(datapath, "Sxx", f"{offset}.mat")
Sxx_GT = sio.loadmat(Sxx_GT_path)['export_Sxx']
Sxx_GT = torch.tensor(Sxx_GT, dtype=torch.complex128, device=device)


Sxy_GT_path = os.path.join(datapath, "Sxy", f"{offset}.mat")
Sxy_GT = sio.loadmat(Sxy_GT_path)['export_Sxy']
Sxy_GT = torch.tensor(Sxy_GT, dtype=torch.complex128, device=device)


Syy_GT_path = os.path.join(datapath, "Syy", f"{offset}.mat")
Syy_GT = sio.loadmat(Syy_GT_path)['export_Syy']
Syy_GT = torch.tensor(Syy_GT, dtype=torch.complex128, device=device)


x_u_GT_path = os.path.join(datapath, "x_u", f"{offset}.mat")
x_u_GT = sio.loadmat(x_u_GT_path)['export_x_u']
x_u_GT = torch.tensor(x_u_GT, dtype=torch.complex128, device=device)

    
x_v_GT_path = os.path.join(datapath, "x_v", f"{offset}.mat")
x_v_GT = sio.loadmat(x_v_GT_path)['export_x_v']
x_v_GT = torch.tensor(x_v_GT, dtype=torch.complex128, device=device)


known_index_rho_water = random_index(500, 128, seed=6)
known_index_p_t = random_index(500, 128, seed=5)
known_index_Sxx = random_index(500, 128, seed=4)
known_index_Sxy = random_index(500, 128, seed=3)
known_index_Syy = random_index(500, 128, seed=2)
known_index_x_u = random_index(500, 128, seed=1)
known_index_x_v = random_index(500, 128, seed=0)



rho_water_N = rho_water_GT.unsqueeze(0).unsqueeze(0)
p_t_N = p_t_GT.unsqueeze(0).unsqueeze(0)
Sxx_N = Sxx_GT.unsqueeze(0).unsqueeze(0)
Sxy_N = Sxy_GT.unsqueeze(0).unsqueeze(0)
Syy_N = Syy_GT.unsqueeze(0).unsqueeze(0)
x_u_N = x_u_GT.unsqueeze(0).unsqueeze(0)
x_v_N = x_v_GT.unsqueeze(0).unsqueeze(0)


pde_loss_AC_real, pde_loss_AC_imag, pde_loss_structure_real_x, pde_loss_structure_imag_x, pde_loss_structure_real_y, pde_loss_structure_imag_y, observation_loss_rho_water, observation_loss_p_t, observation_loss_Sxx, observation_loss_Sxy, observation_loss_Syy, observation_loss_x_u, observation_loss_x_v = get_VA_loss(rho_water_N, p_t_N, Sxx_N, Sxy_N, Syy_N, x_u_N, x_v_N, rho_water_GT, p_t_GT, Sxx_GT, Sxy_GT, Syy_GT, x_u_GT, x_v_GT, known_index_rho_water, known_index_p_t, known_index_Sxx, known_index_Sxy, known_index_Syy, known_index_x_u, known_index_x_v, device=device)
 

L_pde_AC_real = torch.norm(pde_loss_AC_real, 2)/(128*128)
L_pde_AC_imag = torch.norm(pde_loss_AC_imag, 2)/(128*128)
L_pde_AC_structure_real_x = torch.norm(pde_loss_structure_real_x, 2)/(128*128)
L_pde_AC_structure_imag_x = torch.norm(pde_loss_structure_imag_x, 2)/(128*128)
L_pde_AC_structure_real_y = torch.norm(pde_loss_structure_real_y, 2)/(128*128)
L_pde_AC_structure_imag_y = torch.norm(pde_loss_structure_imag_y, 2)/(128*128)


L_obs_rho_water = torch.norm(observation_loss_rho_water, 2)/500
L_obs_p_t = torch.norm(observation_loss_p_t, 2)/500
L_obs_Sxx = torch.norm(observation_loss_Sxx, 2)/500
L_obs_Sxy = torch.norm(observation_loss_Sxy, 2)/500
L_obs_Syy = torch.norm(observation_loss_Syy, 2)/500
L_obs_x_u = torch.norm(observation_loss_x_u, 2)/500
L_obs_x_v = torch.norm(observation_loss_x_v, 2)/500


print(L_pde_AC_real)
print(L_pde_AC_imag)
print(L_pde_AC_structure_real_x)
print(L_pde_AC_structure_imag_x)
print(L_pde_AC_structure_real_y)
print(L_pde_AC_structure_imag_y)


# scipy.io.savemat('L_pde_NS.mat', {'L_pde_NS': L_pde_NS.cpu().detach().numpy()})
# scipy.io.savemat('L_pde_heat.mat', {'L_pde_heat': L_pde_heat.cpu().detach().numpy()})


pde_loss_AC_real = pde_loss_AC_real.cuda()
pde_loss_AC_imag = pde_loss_AC_imag.cuda()

pde_loss_structure_real_x = pde_loss_structure_real_x.cuda()
pde_loss_structure_imag_x = pde_loss_structure_imag_x.cuda()
pde_loss_structure_real_y = pde_loss_structure_real_y.cuda()
pde_loss_structure_imag_y = pde_loss_structure_imag_y.cuda()


pde_loss_AC_real_np = pde_loss_AC_real.cpu().numpy()
pde_loss_AC_imag_np = pde_loss_AC_imag.cpu().numpy()

pde_loss_structure_real_x_np = pde_loss_structure_real_x.cpu().numpy()
pde_loss_structure_imag_x_np = pde_loss_structure_imag_x.cpu().numpy()
pde_loss_structure_real_y_np = pde_loss_structure_real_y.cpu().numpy()
pde_loss_structure_imag_y_np = pde_loss_structure_imag_y.cpu().numpy()

scipy.io.savemat('pde_loss_structure_imag_x.mat', {'pde_loss_structure_imag_x': pde_loss_structure_imag_x.cpu().detach().numpy()})
scipy.io.savemat('pde_loss_structure_imag_y.mat', {'pde_loss_structure_imag_y': pde_loss_structure_imag_y.cpu().detach().numpy()})



plt.figure(figsize=(8, 6))
plt.imshow(pde_loss_AC_real_np, cmap='viridis')  # 使用 viridis 色彩映射
plt.colorbar()  # 添加颜色条
plt.title('VA Loss for AC_real')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('VA _loss_AC_real.png')  # 保存为 PNG 文件

plt.figure(figsize=(8, 6))
plt.imshow(pde_loss_AC_imag_np, cmap='viridis')  # 使用 viridis 色彩映射
plt.colorbar()  # 添加颜色条
plt.title('VA Loss for AC_imag')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('VA_loss_AC_imag.png')  # 保存为 PNG 文件




# 绘制 pde_loss_structure 的图像
plt.figure(figsize=(8, 6))
plt.imshow(pde_loss_structure_real_x_np, cmap='viridis')  # 使用相同的色彩映射
plt.colorbar()
plt.title('VA Loss structure_real_x')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('VA_loss_structure_real_x.png')  # 保存为 PNG 文件


plt.figure(figsize=(8, 6))
plt.imshow(pde_loss_structure_imag_x_np, cmap='viridis')  # 使用相同的色彩映射
plt.colorbar()
plt.title('VA_loss_structure_imag_x')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('VA_loss_structure_imag_x.png')  # 保存为 PNG 文件


plt.figure(figsize=(8, 6))
plt.imshow(pde_loss_structure_real_y_np, cmap='viridis')  # 使用相同的色彩映射
plt.colorbar()
plt.title('PDE Loss structure_real_y')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('VA_loss_structure_real_y.png')  # 保存为 PNG 文件


plt.figure(figsize=(8, 6))
plt.imshow(pde_loss_structure_imag_y_np, cmap='viridis')  # 使用相同的色彩映射
plt.colorbar()
plt.title('VA_loss_structure_imag_y')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('VA_loss_structure_imag_y.png')  # 保存为 PNG 文件
