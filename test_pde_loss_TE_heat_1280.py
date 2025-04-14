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
import math
import matplotlib.pyplot as plt


def identify_mater(elliptic_params, device=torch.device('cuda')):

    # elliptic_params: 椭圆参数，格式为 [长轴, 短轴, 旋转角度（度）]
    mater_iden = torch.zeros(1280, 1280, device=device)
    
    # 椭圆参数
    e_a = elliptic_params[0,0]  # 长轴
    e_b = elliptic_params[0,1]  # 短轴
    angle = math.radians(elliptic_params[0,2])  # 旋转角度，转换为弧度

    center_x = 0
    center_y = 0

    for j in range(1280):
        for k in range(1280):
            x0 = -63.5 + 0.1*j
            y0 = -63.5 + 0.1*k

            # 旋转坐标系，将椭圆旋转到标准位置
            x_rot = (x0 - center_x) * math.cos(angle) + (y0 - center_y) * math.sin(angle)
            y_rot = -(x0 - center_x) * math.sin(angle) + (y0 - center_y) * math.cos(angle)

            # 判断点是否在椭圆内
            if (x_rot / e_a) ** 2 + (y_rot / e_b) ** 2 <= 1:
                mater_iden[j, k] = 1  # 椭圆内部或边界
            else:
                mater_iden[j, k] = -1  # 椭圆外部

    # 保存为 .mat 文件
    # scipy.io.savemat('mater_iden.mat', {'mater_iden': mater_iden.cpu().detach().numpy()})
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


def get_boundary_points(grid_size, device=torch.device('cuda')):
    idx = torch.arange(grid_size, device=device)
    boundary_mask = torch.zeros((grid_size, grid_size), device=device)

    boundary_mask[0, :] = 1      # top
    boundary_mask[-1, :] = 1     # bottom
    boundary_mask[:, 0] = 1      # left
    boundary_mask[:, -1] = 1     # right

    return boundary_mask


def generate_separa_mater(mater, T, mater_iden, device=torch.device('cuda')):
    f = 4e9
    k_0 = 2 * np.pi * f / 3e8
    omega = 2 * np.pi * f
    q = 1.602
    miu_r = 1
    eps_0 = 8.854e-12
    kB = 8.6173e-5
    Eg = 1.12
  
    sigma_coef_map = torch.where(mater_iden > 1e-5, mater, 0)
    sigma_map = q * sigma_coef_map * torch.exp(- Eg / (kB * T))   #与T有关
    sigma_map = torch.where(mater_iden > 1e-5, sigma_map, 1e-7)
    pho_map = torch.where(mater_iden > 1e-5, 70, mater)
    eps_r = torch.where(mater_iden > 1e-5, 11.7, 1)
    K_map = miu_r * k_0**2 * (eps_r - 1j * sigma_map/(omega * eps_0))


    # eps_i=sigma_map/(omega * eps_0)

    # scipy.io.savemat('sigma_map.mat', {'sigma_map': sigma_map.cpu().detach().numpy()})
    # scipy.io.savemat('sigma_coef_map.mat', {'sigma_coef_map': sigma_coef_map.cpu().detach().numpy()})
    # scipy.io.savemat('K_map.mat', {'K_map': K_map.cpu().detach().numpy()})

    # scipy.io.savemat('eps_i.mat', {'eps_i': eps_i.cpu().detach().numpy()})
    # scipy.io.savemat('eps_r.mat', {'eps_r': eps_r.cpu().detach().numpy()})


    return sigma_map, pho_map, K_map


def get_L_BC_heat(T, pho, boundary_mask, device=torch.device('cuda')):

    T_ext = 293.15
    h_trans = 15  # 读取
    T = T.detach().clone().requires_grad_(True)

    delta_x = 128/1280*1e-3 # 1mm
    delta_y = 128/1280*1e-3 # 1mm

    deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * delta_x)
    deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * delta_y)

    dT_dx = F.conv2d(T, deriv_x, padding=(0, 1))
    dT_dy = F.conv2d(T, deriv_y, padding=(1, 0))

    # 边界法向导数：只保留边界方向（这里是硬编码矩形边界）
    dT_dn = torch.zeros_like(dT_dy)  # 和 dT_dy 保持维度一致
    T_s = T.squeeze()
    # 一阶差分
    dT_dn[0, :]  = -(T_s[1, :] - T_s[0, :]) / delta_y       # 上边界 (y=0)
    dT_dn[-1, :] =  (T_s[-1, :] - T_s[-2, :]) / delta_y     # 下边界 (y=127)
    dT_dn[:, 0]  = -(T_s[:, 1] - T_s[:, 0]) / delta_x  # left
    dT_dn[:, -1] =  (T_s[:, -1] - T_s[:, -2]) / delta_x  # right

    T_boundary = T.squeeze()
    result_BC_T = pho * dT_dn + h_trans * (T_boundary - T_ext)
    result_BC_T = result_BC_T * boundary_mask  

    # scipy.io.savemat('result_BC_T.mat', {'result_BC_T': result_BC_T.cpu().detach().numpy()})
    # print(result_BC_T)

    return result_BC_T



def get_TE_heat_loss(mater, Ez, T, mater_GT, Ez_GT, T_GT, mater_mask, Ez_mask, T_mask, boundary_mask, mater_iden, device=torch.device('cuda')):
    """Return the loss of the TE_heat equation and the observation loss."""

    sigma, pho, K_E = generate_separa_mater(mater, T, mater_iden)

    delta_x = 128/1280*1e-3 # 1mm
    delta_y = 128/1280*1e-3 # 1mm
    
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * delta_x)
    deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * delta_y)

    deriv_x_complex = torch.complex(deriv_x, torch.zeros_like(deriv_x))
    deriv_y_complex = torch.complex(deriv_y, torch.zeros_like(deriv_y))
    

    # E_filed
    grad_x_next_x_E = F.conv2d(Ez, deriv_x_complex, padding=(0, 1))
    grad_x_next_y_E = F.conv2d(Ez, deriv_y_complex, padding=(1, 0))
    Laplac_E = F.conv2d(grad_x_next_x_E, deriv_x_complex, padding=(0, 1)) + F.conv2d(grad_x_next_y_E, deriv_y_complex, padding=(1, 0))
    result_E = Laplac_E + K_E * Ez

    part_E = K_E * Ez

    scipy.io.savemat('K_E.mat', {'K_E': K_E.cpu().detach().numpy()})
    scipy.io.savemat('Laplac_E.mat', {'Laplac_E': Laplac_E.cpu().detach().numpy()})
    scipy.io.savemat('part_E.mat', {'part_E': part_E.cpu().detach().numpy()})
    scipy.io.savemat('result_E.mat', {'result_E': result_E.cpu().detach().numpy()})

    # T_filed
    grad_x_next_x_T = F.conv2d(T, deriv_x, padding=(0, 1))
    grad_x_next_y_T = F.conv2d(T, deriv_y, padding=(1, 0))
    Laplac_T = F.conv2d(grad_x_next_x_T, deriv_x, padding=(0, 1)) + F.conv2d(grad_x_next_y_T, deriv_y, padding=(1, 0))
    result_T = pho * Laplac_T + 0.5 * sigma * Ez * torch.conj(Ez)
    # scipy.io.savemat('Laplac_T.mat', {'Laplac_T': Laplac_T.cpu().detach().numpy()})

    # Robin_T
    result_BC_T = get_L_BC_heat(T, pho, boundary_mask, device=device)


    pde_loss_E = result_E
    pde_loss_T = result_T
    pde_loss_E = pde_loss_E.squeeze()
    pde_loss_T = pde_loss_T.squeeze()
    
    pde_loss_BC_T = result_BC_T
    pde_loss_BC_T = pde_loss_BC_T.squeeze()

    # scipy.io.savemat('pde_loss_E.mat', {'pde_loss_E': pde_loss_E.cpu().detach().numpy()})
    # scipy.io.savemat('pde_loss_T.mat', {'pde_loss_T': pde_loss_T.cpu().detach().numpy()})
    # scipy.io.savemat('result_E.mat', {'result_E': result_E.cpu().detach().numpy()})
    # scipy.io.savemat('Laplac_E.mat', {'Laplac_E': Laplac_E.cpu().detach().numpy()})
    # scipy.io.savemat('result_T.mat', {'result_T': result_T.cpu().detach().numpy()})


    observation_loss_mater = (mater - mater_GT).squeeze()
    observation_loss_mater = observation_loss_mater * mater_mask  
    observation_loss_Ez = (Ez - Ez_GT).squeeze()
    observation_loss_Ez = observation_loss_Ez * Ez_mask  
    observation_loss_T = (T - T_GT).squeeze()
    observation_loss_T = observation_loss_T * T_mask


    # scipy.io.savemat('observation_loss_mater.mat', {'observation_loss_mater': observation_loss_mater.cpu().detach().numpy()})
    # scipy.io.savemat('observation_loss_Ez.mat', {'observation_loss_Ez': observation_loss_Ez.cpu().detach().numpy()})
    # scipy.io.savemat('observation_loss_T.mat', {'observation_loss_T': observation_loss_T.cpu().detach().numpy()})

    pde_loss_E = pde_loss_E/1000000
    pde_loss_T = pde_loss_T/1000000

    pde_loss_BC_T = pde_loss_BC_T/1000

    return pde_loss_E, pde_loss_T, pde_loss_BC_T, observation_loss_mater, observation_loss_Ez, observation_loss_T




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datapath = '/data/yangchangfan/DiffusionPDE/data/testing/TE_heat'
offset = 1

mater_GT_path = os.path.join(datapath, "mater", f"{offset}.mat")
mater_GT = sio.loadmat(mater_GT_path)['mater']
mater_GT = torch.tensor(mater_GT, dtype=torch.float64, device=device)

Ez_GT_path = os.path.join(datapath, "Ez", f"{offset}.mat")
Ez_GT = sio.loadmat(Ez_GT_path)['export_Ez']
Ez_GT = torch.tensor(Ez_GT, dtype=torch.complex128, device=device)
    
T_GT_path = os.path.join(datapath, "T", f"{offset}.mat")
T_GT = sio.loadmat(T_GT_path)['export_T']
T_GT = torch.tensor(T_GT, dtype=torch.float64, device=device)

elliptic_GT_path = os.path.join(datapath, "ellipticcsv", f"{offset}.csv")
elliptic_GT = pd.read_csv(elliptic_GT_path, header=None)
elliptic_GT = torch.tensor(elliptic_GT.values, dtype=torch.float64)

known_index_mater = random_index(500, 1280, seed=2)
known_index_Ez = random_index(500, 1280, seed=1)
known_index_T = random_index(500, 1280, seed=0)
boundary_mask = get_boundary_points(1280,device=device)

mater_in = (mater_GT >= 1e11) & (mater_GT <= 3e11)
mater_out = (mater_GT >= 10) & (mater_GT <= 20)
normal_datamater = torch.where(mater_in, (mater_GT - 1e11) / (3e11 - 1e11) * 0.8 + 0.1, (mater_GT - 10) / (20 - 10) * 0.8 - 0.9)
    # 边上和内部设置为parm.Sigma_Si_coef(0.1,0.9)，其他设置为normal_Pho_Al(-0.9,-0.1)

mater_N = normal_datamater.unsqueeze(0).unsqueeze(0)
scipy.io.savemat('mater1.mat', {'mater_N': mater_N.cpu().detach().numpy()})

mater_iden = identify_mater(elliptic_GT)
scipy.io.savemat('mater_iden_test.mat', {'mater_iden ': mater_iden .cpu().detach().numpy()})

val_in = ((mater_N - 0.1) * (3e11 - 1e11) / 0.8 + 1e11).to(torch.float64)  
val_out = ((mater_N + 0.9) * (20 - 10) / 0.8 + 10).to(torch.float64)  
mater_N = torch.where(mater_iden > 1e-5, val_in, val_out)
scipy.io.savemat('mater2.mat', {'mater_N': mater_N.cpu().detach().numpy()})

Ez_N = Ez_GT.unsqueeze(0).unsqueeze(0)
T_N = T_GT.unsqueeze(0).unsqueeze(0)  



pde_loss_E, pde_loss_T, pde_loss_BC_T, observation_loss_mater, observation_loss_Ez, observation_loss_T = get_TE_heat_loss(mater_N, Ez_N, T_N, mater_GT, Ez_GT, T_GT, known_index_mater, known_index_Ez, known_index_T, boundary_mask, mater_iden, device=device)

L_pde_E = torch.norm(pde_loss_E, 2)/(1280*1280)
L_pde_T = torch.norm(pde_loss_T, 2)/(1280*1280)
L_obs_mater = torch.norm(observation_loss_mater, 2)/500
L_obs_Ez = torch.norm(observation_loss_Ez, 2)/500
L_obs_T = torch.norm(observation_loss_T, 2)/500

L_obs_BC_T = torch.norm(pde_loss_BC_T, 2)/(1280+1280+1280+1280-4)

print(L_pde_E)
print(L_pde_T)
print(L_obs_BC_T)


# scipy.io.savemat('L_pde_NS.mat', {'L_pde_NS': L_pde_NS.cpu().detach().numpy()})
# scipy.io.savemat('L_pde_heat.mat', {'L_pde_heat': L_pde_heat.cpu().detach().numpy()})


pde_loss_E = pde_loss_E.cuda()
pde_loss_T = pde_loss_T.cuda()

pde_loss_E_np = pde_loss_E.cpu().numpy()
pde_loss_T_np = pde_loss_T.cpu().numpy()

# vmin_pde_loss_E, vmax_pde_loss_E = pde_loss_E.min(), pde_loss_E.max()
# vmin_pde_loss_T, vmax_pde_loss_T = pde_loss_T.min(), pde_loss_T.max()

plt.figure(figsize=(8, 6))
plt.imshow(pde_loss_E_np.real, cmap='viridis', vmin=-0.1, vmax=0.1)  # 使用 viridis 色彩映射
plt.colorbar()  # 添加颜色条
plt.title('TE_heat Loss for E')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('TE_heat_loss_E.png')  # 保存为 PNG 文件


# 绘制 pde_loss_T 的图像
plt.figure(figsize=(8, 6))
plt.imshow(pde_loss_T_np.real, cmap='viridis', vmin=-0.01, vmax=0.01)  # 使用相同的色彩映射
plt.colorbar()
plt.title('TE_heat Loss for T')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('TE_heat_loss_T.png')  # 保存为 PNG 文件