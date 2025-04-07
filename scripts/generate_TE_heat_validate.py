import tqdm
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import scipy.io
import os
import scipy.io as sio
import pandas as pd

import numpy as np
from shapely.geometry import Polygon, Point
import json


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


def identify_mater(poly_GT, device=torch.device('cuda')):
    mater_iden = torch.zeros(128,128, device=device)
    polygon = Polygon(poly_GT)

    for j in range(128):
       for k in range(128):
           x0 = -63.5 + j
           y0 = -63.5 + k
           point = Point(x0, y0)  
           if polygon.contains(point) or polygon.boundary.contains(point):
                mater_iden[j, k] = 1
           else:
                mater_iden[j, k] = -1
    return mater_iden
    

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


    scipy.io.savemat('sigma_map.mat', {'sigma_map': sigma_map.cpu().detach().numpy()})
    scipy.io.savemat('sigma_coef_map.mat', {'sigma_coef_map': sigma_coef_map.cpu().detach().numpy()})
    scipy.io.savemat('T.mat', {'T': T.cpu().detach().numpy()})

    return sigma_map, pho_map, K_map


def get_TE_heat_loss(mater, Ez, T, mater_GT, Ez_GT, T_GT, mater_mask, Ez_mask, T_mask, mater_iden, device=torch.device('cuda')):
    """Return the loss of the TE_heat equation and the observation loss."""

    sigma, pho, K_E = generate_separa_mater(mater, T, mater_iden)

    delta_x = 1e-3 # 1mm
    delta_y = 1e-3 # 1mm
    
    deriv_x = torch.tensor([[1, 0, -1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * delta_x)
    deriv_y = torch.tensor([[1], [0], [-1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * delta_y)

    deriv_x_complex = torch.complex(deriv_x, torch.zeros_like(deriv_x))
    deriv_y_complex = torch.complex(deriv_y, torch.zeros_like(deriv_y))

    # E_filed
    grad_x_next_x_E = F.conv2d(Ez, deriv_x_complex, padding=(0, 1))
    grad_x_next_y_E = F.conv2d(Ez, deriv_y_complex, padding=(1, 0))
    Laplac_E = F.conv2d(grad_x_next_x_E, deriv_x_complex, padding=(0, 1)) + F.conv2d(grad_x_next_y_E, deriv_y_complex, padding=(1, 0))
    result_E = Laplac_E + K_E * Ez
    # T_filed
    grad_x_next_x_T = F.conv2d(T, deriv_x, padding=(0, 1))
    grad_x_next_y_T = F.conv2d(T, deriv_y, padding=(1, 0))
    Laplac_T = F.conv2d(grad_x_next_x_T, deriv_x, padding=(0, 1)) + F.conv2d(grad_x_next_y_T, deriv_y, padding=(1, 0))
    result_T = pho * Laplac_T + 0.5 * sigma * Ez * torch.conj(Ez)

    pde_loss_E = result_E
    pde_loss_T = result_T

    pde_loss_E = pde_loss_E.squeeze()
    pde_loss_T = pde_loss_T.squeeze()
    
    scipy.io.savemat('pde_loss_E.mat', {'pde_loss_E': pde_loss_E.cpu().detach().numpy()})
    scipy.io.savemat('pde_loss_T.mat', {'pde_loss_T': pde_loss_T.cpu().detach().numpy()})
    scipy.io.savemat('result_E.mat', {'result_E': result_E.cpu().detach().numpy()})
    scipy.io.savemat('Laplac_E.mat', {'Laplac_E': Laplac_E.cpu().detach().numpy()})
    scipy.io.savemat('result_T.mat', {'result_T': result_T.cpu().detach().numpy()})


    observation_loss_mater = (mater - mater_GT).squeeze()
    observation_loss_mater = observation_loss_mater * mater_mask  
    observation_loss_Ez = (Ez - Ez_GT).squeeze()
    observation_loss_Ez = observation_loss_Ez * Ez_mask  
    observation_loss_T = (T - T_GT).squeeze()
    observation_loss_T = observation_loss_T * T_mask

    return pde_loss_E, pde_loss_T, observation_loss_mater, observation_loss_Ez, observation_loss_T

def save_batch_results(results, outdir, batch_indices):
    """批量保存结果，并返回误差列表"""
    batch_errors_mater = []
    batch_errors_Ez = []
    batch_errors_T = []
    
    for idx, result in zip(batch_indices, results):
        # 保存每个样本的结果
        results_dir = os.path.join(outdir, f'sample_{idx}')
        os.makedirs(results_dir, exist_ok=True)
        
        scipy.io.savemat(os.path.join(results_dir, 'TE_heat_results.mat'), {
            'mater': result['mater'],
            'Ez': result['Ez'],
            'T': result['T']
        })
        
        # 收集误差
        batch_errors_mater.append(result['relative_errors']['mater'])
        batch_errors_Ez.append(result['relative_errors']['Ez'])
        batch_errors_T.append(result['relative_errors']['T'])
    
    return batch_errors_mater, batch_errors_Ez, batch_errors_T


def load_batch_data(datapath, batch_indices, device):
    """批量加载数据"""
    batch_data = []
    for idx in batch_indices:
        mater_GT = sio.loadmat(os.path.join(datapath, "mater", f"{idx}.mat"))['mater']
        Ez_GT = sio.loadmat(os.path.join(datapath, "Ez", f"{idx}.mat"))['export_Ez']
        T_GT = sio.loadmat(os.path.join(datapath, "T", f"{idx}.mat"))['export_T']
        poly_GT = pd.read_csv(os.path.join(datapath, "polycsv", f"{idx}.csv"), header=None).values
        
        batch_data.append({
            'mater_GT': torch.tensor(mater_GT, dtype=torch.float64, device=device),
            'Ez_GT': torch.tensor(Ez_GT, dtype=torch.complex128, device=device),
            'T_GT': torch.tensor(T_GT, dtype=torch.float64, device=device),
            'poly_GT': torch.tensor(poly_GT, dtype=torch.float64)
        })
    return batch_data




def process_batch(batch_data, net, config):
    """批量处理数据"""
    results = []
    device = config['generate']['device']
    
    for data in batch_data:
        # 获取当前样本的数据
        mater_GT = data['mater_GT']
        Ez_GT = data['Ez_GT']
        T_GT = data['T_GT']
        poly_GT = data['poly_GT']
        
        # 预先计算mater_iden，只计算一次
        mater_iden = identify_mater(poly_GT)
        
        # 设置EDM参数
        batch_size = config['generate']['batch_size']
        latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        
        # 设置采样参数
        num_steps = config['test']['iterations']
        sigma_min = config['generate']['sigma_min']
        sigma_max = config['generate']['sigma_max']
        rho = config['generate']['rho']
        
        # 设置sigma步长
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        sigma_t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        sigma_t_steps = torch.cat([net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])])
        
        # 初始化
        x_next = latents.to(torch.float64) * sigma_t_steps[0]
        known_index_mater = random_index(500, 128, seed=2)
        known_index_Ez = random_index(500, 128, seed=1)
        known_index_T = random_index(500, 128, seed=0)
        
        # EDM采样和优化循环
        for i, (sigma_t_cur, sigma_t_next) in enumerate(zip(sigma_t_steps[:-1], sigma_t_steps[1:])):
            x_cur = x_next.detach().clone()
            x_cur.requires_grad = True
            sigma_t = net.round_sigma(sigma_t_cur)
            
            # Euler step
            x_N = net(x_cur, sigma_t).to(torch.float64)
            d_cur = (x_cur - x_N) / sigma_t
            x_next = x_cur + (sigma_t_next - sigma_t) * d_cur
            
            # 计算损失和梯度
            mater_N = x_N[:,0,:,:].unsqueeze(0)
            real_Ez_N = x_N[:,1,:,:].unsqueeze(0)
            imag_Ez_N = x_N[:,2,:,:].unsqueeze(0)
            T_N = x_N[:,3,:,:].unsqueeze(0)
            
            
            # inv_normalization
            val_in = ((mater_N - 0.1) * (3e11 - 1e11) / 0.8 + 1e11).to(torch.float64)
            val_out = ((mater_N + 0.9) * (20 - 10) / 0.8 + 10).to(torch.float64)
            mater_N = torch.where(mater_iden > 1e-5, val_in, val_out)

            # 加载归一化参数
            max_abs_Ez_path = "/data/yangchangfan/DiffusionPDE/data/training/TE_heat/Ez/max_abs_Ez.mat"
            max_abs_Ez = sio.loadmat(max_abs_Ez_path)['max_abs_Ez']
            max_abs_Ez = torch.tensor(max_abs_Ez, device=device)

            range_allT_paths = "/data/yangchangfan/DiffusionPDE/data/training/TE_heat/T/range_allT.mat"
            range_allT = sio.loadmat(range_allT_paths)['range_allT']
            range_allT = torch.tensor(range_allT, device=device)

            max_T = range_allT[0,1]
            min_T = range_allT[0,0]

            # 反归一化
            real_Ez_N = (real_Ez_N*max_abs_Ez/0.9).to(torch.float64)
            imag_Ez_N = (imag_Ez_N*max_abs_Ez/0.9).to(torch.float64)
            T_N = ((T_N+0.9)/1.8 *(max_T - min_T) + min_T).to(torch.float64)

            # 构建复数Ez
            complex_Ez_N = torch.complex(real_Ez_N, imag_Ez_N)

            
            # 计算损失 - 使用预计算的 
            pde_loss_E, pde_loss_T, obs_loss_mater, obs_loss_Ez, obs_loss_T = get_TE_heat_loss(mater_N, complex_Ez_N, T_N, mater_GT, Ez_GT, T_GT, known_index_mater, known_index_Ez, known_index_T, mater_iden, device)
            
            # 计算损失的L2范数
            L_pde_E = torch.norm(pde_loss_E, 2)/(128*128)
            L_pde_T = torch.norm(pde_loss_T, 2)/(128*128)
            L_obs_mater = torch.norm(obs_loss_mater, 2)/500
            L_obs_Ez = torch.norm(obs_loss_Ez, 2)/500
            L_obs_T = torch.norm(obs_loss_T, 2)/500

            # 计算梯度
            grad_x_cur_obs_mater = torch.autograd.grad(L_obs_mater, x_cur, retain_graph=True)[0]
            grad_x_cur_obs_Ez = torch.autograd.grad(L_obs_Ez, x_cur, retain_graph=True)[0]
            grad_x_cur_obs_T = torch.autograd.grad(L_obs_T, x_cur, retain_graph=True)[0]
            grad_x_cur_pde_E = torch.autograd.grad(L_pde_E, x_cur, retain_graph=True)[0]
            grad_x_cur_pde_T = torch.autograd.grad(L_pde_T, x_cur)[0]

            # 设置zeta参数
            zeta_obs_mater = 10
            zeta_obs_Ez = 10
            zeta_obs_T = 10
            zeta_pde_E = 10
            zeta_pde_T = 10

            # 归一化梯度
            norm_mater = torch.norm(zeta_obs_mater * grad_x_cur_obs_mater)
            scale_factor = 1.0 / norm_mater
            zeta_obs_mater = zeta_obs_mater * scale_factor

            norm_Ez = torch.norm(zeta_obs_Ez * grad_x_cur_obs_Ez)
            scale_factor = 1.0 / norm_Ez
            zeta_obs_Ez = zeta_obs_Ez * scale_factor

            norm_T = torch.norm(zeta_obs_T * grad_x_cur_obs_T)
            scale_factor = 1.0 / norm_T
            zeta_obs_T = zeta_obs_T * scale_factor

            # 更新x_next
            if i <= 0.9 * num_steps:
                x_next = x_next - zeta_obs_mater * grad_x_cur_obs_mater - zeta_obs_Ez * grad_x_cur_obs_Ez - zeta_obs_T * grad_x_cur_obs_T
            else:
                norm_pde_E = torch.norm(zeta_pde_E * grad_x_cur_pde_E)
                scale_factor = 1 / norm_pde_E
                zeta_pde_E = zeta_pde_E * scale_factor

                norm_pde_T = torch.norm(zeta_pde_T * grad_x_cur_pde_E)
                scale_factor = 1 / norm_pde_T
                zeta_pde_T = zeta_pde_T * scale_factor

                x_next = x_next - 0.95 * (zeta_obs_mater * grad_x_cur_obs_mater + zeta_obs_Ez * grad_x_cur_obs_Ez + zeta_obs_T * grad_x_cur_obs_T) - 0.05 * (zeta_pde_E * grad_x_cur_pde_E + zeta_pde_T * grad_x_cur_pde_T)
        
        # 处理最终结果
        x_final = x_next
        mater_final = x_final[:,0,:,:].unsqueeze(0)
        real_Ez_final = x_final[:,1,:,:].unsqueeze(0)
        imag_Ez_final = x_final[:,2,:,:].unsqueeze(0)
        T_final = x_final[:,3,:,:].unsqueeze(0)
        
        mater_iden = identify_mater(poly_GT)
        val_in = ((mater_final - 0.1) * (3e11 - 1e11) / 0.8 + 1e11).to(torch.float64)  
        val_out = ((mater_final + 0.9) * (20 - 10) / 0.8 + 10).to(torch.float64)  
        mater_final = torch.where(mater_iden > 1e-5, val_in, val_out)


        real_Ez_final = (real_Ez_final * max_abs_Ez / 0.9).to(torch.float64)
        imag_Ez_final = (imag_Ez_final * max_abs_Ez / 0.9).to(torch.float64)
        complex_Ez_final = torch.complex(real_Ez_final, imag_Ez_final)
        T_final = ((T_final+0.9)/1.8 *(max_T - min_T) + min_T).to(torch.float64)
        
        # 计算相对误差
        relative_error_mater = torch.norm(mater_final - mater_GT, 2) / torch.norm(mater_GT, 2)
        relative_error_Ez = torch.norm(complex_Ez_final - Ez_GT, 2) / torch.norm(Ez_GT, 2)
        relative_error_T = torch.norm(T_final - T_GT, 2) / torch.norm(T_GT, 2)
        
        result = {
            'mater': mater_final.cpu().detach().numpy(),
            'Ez': complex_Ez_final.cpu().detach().numpy(),
            'T': T_final.cpu().detach().numpy(),
            'relative_errors': {
                'mater': relative_error_mater.item(),
                'Ez': relative_error_Ez.item(),
                'T': relative_error_T.item()
            }
        }
        results.append(result)
    
    return results


def generate_TE_heat_validate(config):
    """Generate TE_heat equation."""
    datapath = config['data']['datapath']
    start_idx = config['data']['start_idx']      # 起始索引
    num_samples = config['data']['num_samples']   # 样本数量
    device = config['generate']['device']
    
    # 存储所有样本的相对误差
    all_relative_errors_mater = []
    all_relative_errors_Ez = []
    all_relative_errors_T = []
    
    # 加载预训练模型
    network_pkl = config['test']['pre-trained']
    print(f'Loading networks from "{network_pkl}"...')
    with open(network_pkl, 'rb') as f:
        net = pickle.load(f)['ema'].to(device)
    
    # 批量处理样本
    batch_size = config['generate']['batch_size']
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_indices = range(start_idx + batch_start, start_idx + batch_end)
        
        # 批量处理和保存结果
        batch_data = load_batch_data(datapath, batch_indices, device)
        results = process_batch(batch_data, net, config)
        batch_errors = save_batch_results(results, config['generate']['outdir'], batch_indices)
        all_relative_errors_mater.extend(batch_errors[0])
        all_relative_errors_Ez.extend(batch_errors[1])
        all_relative_errors_T.extend(batch_errors[2])
        
        # 修正的进度显示
        print(f"Processed samples {batch_start+1} to {batch_end} of {num_samples}")

        # 计算并打印平均误差
        avg_error_mater = np.mean(all_relative_errors_mater)
        avg_error_Ez = np.mean(all_relative_errors_Ez)
        avg_error_T = np.mean(all_relative_errors_T)

                # 保存详细的验证结果到JSON文件
        with open(os.path.join(config['generate']['outdir'], 'validation_results.json'), 'w') as f:
             json.dump({
                'per_sample_errors': {
                    'mater': all_relative_errors_mater,
                    'Ez': all_relative_errors_Ez,
                    'T': all_relative_errors_T
                },
                'statistics': {
                    'average_errors': {
                    'mater': avg_error_mater,
                    'Ez': avg_error_Ez,
                    'T': avg_error_T
                   },
                    }
        }, f, indent=4)
    

    
    print(f'Average Relative Errors:')
    print(f'  Material: {avg_error_mater:.6f} ')
    print(f'  Electric Field: {avg_error_Ez:.6f} ')
    print(f'  Temperature: {avg_error_T:.6f} ')
    
    # 保存所有统计结果
    stats = {
        'mater_errors': all_relative_errors_mater,
        'Ez_errors': all_relative_errors_Ez,
        'T_errors': all_relative_errors_T,
        'avg_error_mater': avg_error_mater,
        'avg_error_Ez': avg_error_Ez,
        'avg_error_T': avg_error_T,
    }
    
    scipy.io.savemat(os.path.join(config['generate']['outdir'], 'validation_stats.mat'), stats)
    

    
    print('Validation completed. Results saved.')

