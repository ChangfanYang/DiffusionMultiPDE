import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import torch

def evaluate(result, GT_data):
    """
    输入:
        result: dict，预测结果，包含 'rho_water', 'real_p_t', 'imag_p_t', 'real_Sxx', 'imag_Sxx', 'real_Sxy', 'imag_Sxy', 'real_Syy', 'imag_Syy', 'real_x_u', 'imag_x_u', 'real_x_v', 'imag_x_v'
        GT_data: dict，GT数据，包含 'rho_water', 'real_p_t', 'imag_p_t', 'real_Sxx', 'imag_Sxx', 'real_Sxy', 'imag_Sxy', 'real_Syy', 'imag_Syy', 'real_x_u', 'imag_x_u', 'real_x_v', 'imag_x_v'
    输出:
        res_dict: dict，包含 RMSE, nRMSE, MaxError, bRMSE, fRMSE 指标
    """
    res_dict = {
        'RMSE': {}, 'nRMSE': {}, 'MaxError': {}, 'bRMSE': {}, 'fRMSE': {}
    }

    key_map = {
        'rho_water': ('rho_water', 'rho_water_GT'),
        'real_p_t': ('real_p_t', 'real_p_t_GT'),
        'imag_p_t': ('imag_p_t', 'imag_p_t_GT'),
        'real_Sxx': ('real_Sxx', 'real_Sxx_GT'),
        'imag_Sxx': ('imag_Sxx', 'imag_Sxx_GT'),
        'real_Sxy': ('real_Sxy', 'real_Sxy_GT'),
        'imag_Sxy': ('imag_Sxy', 'imag_Sxy_GT'),
        'real_Syy': ('real_Syy', 'real_Syy_GT'),
        'imag_Syy': ('imag_Syy', 'imag_Syy_GT'),
        'real_x_u': ('real_x_u', 'real_x_u_GT'),
        'imag_x_u': ('imag_x_u', 'imag_x_u_GT'),
        'real_x_v': ('real_x_v', 'real_x_v_GT'),
        'imag_x_v': ('imag_x_v', 'imag_x_v_GT'),

    }

    for var_name, (pred_key, gt_key) in key_map.items():
        # 直接转换为 tensor
        pred = result[pred_key] if isinstance(result[pred_key], torch.Tensor) else torch.tensor(result[pred_key], dtype=torch.float32)
        gt = GT_data[gt_key] if isinstance(GT_data[gt_key], torch.Tensor) else torch.tensor(GT_data[gt_key], dtype=torch.float32)

        # squeeze 去除多余的维度
        pred = pred.squeeze()
        gt = gt.squeeze()

        # nRMSE

        nrmse = torch.norm(pred - gt,2) / torch.norm(gt,2)
        res_dict['nRMSE'][var_name] = nrmse.item()

        # RMSE
        rmse = torch.sqrt(torch.mean((pred - gt) ** 2)).item()
        res_dict['RMSE'][var_name] = rmse

        # MaxError
        maxerr = torch.max(torch.abs(pred - gt)).item()
        res_dict['MaxError'][var_name] = maxerr

        # bRMSE（边界）
        boundary_mask = torch.zeros_like(gt, dtype=torch.bool)
        boundary_mask[0, :] = True
        boundary_mask[-1, :] = True
        boundary_mask[:, 0] = True
        boundary_mask[:, -1] = True
        pred_b = pred[boundary_mask]
        gt_b = gt[boundary_mask]
        brmse = torch.sqrt(torch.mean((pred_b - gt_b) ** 2)).item()
        res_dict['bRMSE'][var_name] = brmse

        # fRMSE（频率段误差）
        freq_bands = {
            'low': (0, 4),
            'middle': (5, 12),
            'high': (13, None),
        }

        pred_fft = torch.fft.fft2(pred)
        gt_fft = torch.fft.fft2(gt)
        H, W = pred.shape[-2:]

        # 计算频率图像中的径向波数
        kx = torch.fft.fftfreq(H, d=1).to(pred.device).reshape(-1, 1).expand(H, W)
        ky = torch.fft.fftfreq(W, d=1).to(pred.device).reshape(1, -1).expand(H, W)
        radius = torch.sqrt(kx ** 2 + ky ** 2) * max(H, W)

        fRMSE_var = {}
        for band, (k_min, k_max) in freq_bands.items():
            if k_max is None:
                mask = (radius >= k_min)
                k_max = max(H // 2, W // 2)
            else:
                mask = (radius >= k_min) & (radius <= k_max)

            diff_fft = torch.abs(pred_fft - gt_fft) ** 2
            band_error = diff_fft[mask].mean().sqrt()
            fRMSE_var[band] = band_error.item()

        res_dict['fRMSE'][var_name] = fRMSE_var

    return res_dict

# 主程序修改部分
data_path = '/data/yangchangfan/DiffusionPDE/data/testing/VA'
results_path = '/home/yangchangfan/CODE/DiffusionPDE/VA_result'

# 初始化存储所有结果的字典
all_results = {
    'RMSE': {'rho_water': [], 'real_p_t': [], 'imag_p_t': [], 'real_Sxx': [], 'imag_Sxx': [], 'real_Sxy': [], 'imag_Sxy': [], 'real_Syy': [], 'imag_Syy': [], 'real_x_u': [], 'imag_x_u': [], 'real_x_v': [], 'imag_x_v': []},
    'nRMSE': {'rho_water': [], 'real_p_t': [], 'imag_p_t': [], 'real_Sxx': [], 'imag_Sxx': [], 'real_Sxy': [], 'imag_Sxy': [], 'real_Syy': [], 'imag_Syy': [], 'real_x_u': [], 'imag_x_u': [], 'real_x_v': [], 'imag_x_v': []},
    'MaxError': {'rho_water': [], 'real_p_t': [], 'imag_p_t': [], 'real_Sxx': [], 'imag_Sxx': [], 'real_Sxy': [], 'imag_Sxy': [], 'real_Syy': [], 'imag_Syy': [], 'real_x_u': [], 'imag_x_u': [], 'real_x_v': [], 'imag_x_v': []},
    'bRMSE': {'rho_water': [], 'real_p_t': [], 'imag_p_t': [], 'real_Sxx': [], 'imag_Sxx': [], 'real_Sxy': [], 'imag_Sxy': [], 'real_Syy': [], 'imag_Syy': [], 'real_x_u': [], 'imag_x_u': [], 'real_x_v': [], 'imag_x_v': []},
    'fRMSE': {'rho_water': {'low': [], 'middle': [], 'high': []}, 
              'real_p_t': {'low': [], 'middle': [], 'high': []},
              'imag_p_t': {'low': [], 'middle': [], 'high': []},
              'real_Sxx': {'low': [], 'middle': [], 'high': []},
              'imag_Sxx': {'low': [], 'middle': [], 'high': []},
              'real_Sxy': {'low': [], 'middle': [], 'high': []},
              'imag_Sxy': {'low': [], 'middle': [], 'high': []},
              'real_Syy': {'low': [], 'middle': [], 'high': []},
              'imag_Syy': {'low': [], 'middle': [], 'high': []},
              'real_x_u': {'low': [], 'middle': [], 'high': []},
              'imag_x_u': {'low': [], 'middle': [], 'high': []},
              'real_x_v': {'low': [], 'middle': [], 'high': []},
              'imag_x_v': {'low': [], 'middle': [], 'high': []}
              }
}

offset_range=[10001, 11001]
for idx in range(offset_range[0], offset_range[1]):
    try:
        # 加载预测结果
        pred = sio.loadmat(f'{results_path}/VA_results_{idx}.mat')
        rho_water = pred['rho_water']
        p_t = pred['p_t']
        Sxx = pred['Sxx']
        Sxy = pred['Sxy']
        Syy = pred['Syy']
        x_u = pred['x_u']
        x_v = pred['x_v']

        VA_results = {
            'rho_water': rho_water,
            'real_p_t': p_t.real,
            'imag_p_t': p_t.imag,
            'real_Sxx': Sxx.real,
            'imag_Sxx': Sxx.imag,
            'real_Sxy': Sxy.real,
            'imag_Sxy': Sxy.imag,
            'real_Syy': Syy.real,
            'imag_Syy': Syy.imag,
            'real_x_u': x_u.real,
            'imag_x_u': x_u.imag,
            'real_x_v': x_v.real,
            'imag_x_v': x_v.imag,
        }

        # 加载GT数据
        VA_GT = {
            'rho_water_GT': sio.loadmat(f'{data_path}/rho_water/{idx}.mat')['export_rho_water'],
            'real_p_t_GT': sio.loadmat(f'{data_path}/p_t/{idx}.mat')['export_p_t'].real,    
            'imag_p_t_GT': sio.loadmat(f'{data_path}/p_t/{idx}.mat')['export_p_t'].imag,    
            'real_Sxx_GT': sio.loadmat(f'{data_path}/Sxx/{idx}.mat')['export_Sxx'].real,
            'imag_Sxx_GT': sio.loadmat(f'{data_path}/Sxx/{idx}.mat')['export_Sxx'].imag,
            'real_Sxy_GT': sio.loadmat(f'{data_path}/Sxy/{idx}.mat')['export_Sxy'].real,
            'imag_Sxy_GT': sio.loadmat(f'{data_path}/Sxy/{idx}.mat')['export_Sxy'].imag,
            'real_Syy_GT': sio.loadmat(f'{data_path}/Syy/{idx}.mat')['export_Syy'].real,
            'imag_Syy_GT': sio.loadmat(f'{data_path}/Syy/{idx}.mat')['export_Syy'].imag,
            'real_x_u_GT': sio.loadmat(f'{data_path}/x_u/{idx}.mat')['export_x_u'].real,
            'imag_x_u_GT': sio.loadmat(f'{data_path}/x_u/{idx}.mat')['export_x_u'].imag,
            'real_x_v_GT': sio.loadmat(f'{data_path}/x_v/{idx}.mat')['export_x_v'].real,
            'imag_x_v_GT': sio.loadmat(f'{data_path}/x_v/{idx}.mat')['export_x_v'].imag
        }
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        # print(NS_heat_GT['Q_heat_GT'])
        # print(NS_heat_results['Q_heat'])
        # exit()
        # 评估当前数据
        res_dict = evaluate(VA_results, VA_GT)
        
        # 保存所有结果
        for metric in ['RMSE', 'nRMSE', 'MaxError', 'bRMSE']:
            for var in ['rho_water', 'real_p_t', 'imag_p_t', 'real_Sxx', 'imag_Sxx', 'real_Sxy', 'imag_Sxy', 'real_Syy', 'imag_Syy', 'real_x_u', 'imag_x_u', 'real_x_v', 'imag_x_v']:
                all_results[metric][var].append(res_dict[metric][var])
        
        for var in ['rho_water', 'real_p_t', 'imag_p_t', 'real_Sxx', 'imag_Sxx', 'real_Sxy', 'imag_Sxy', 'real_Syy', 'imag_Syy', 'real_x_u', 'imag_x_u', 'real_x_v', 'imag_x_v']:
            for band in ['low', 'middle', 'high']:
                all_results['fRMSE'][var][band].append(res_dict['fRMSE'][var][band])
                
    except Exception as e:
        print(f'跳过 {idx}.mat (错误: {str(e)})')
        continue

# 计算平均误差
avg_results = {
    'RMSE': {}, 'nRMSE': {}, 'MaxError': {}, 'bRMSE': {}, 'fRMSE': {}
}

# 计算普通指标的平均值
for metric in ['RMSE', 'nRMSE', 'MaxError', 'bRMSE']:
    for var in ['rho_water', 'real_p_t', 'imag_p_t', 'real_Sxx', 'imag_Sxx', 'real_Sxy', 'imag_Sxy', 'real_Syy', 'imag_Syy', 'real_x_u', 'imag_x_u', 'real_x_v', 'imag_x_v']:
        avg_results[metric][var] = np.mean(all_results[metric][var])

# 计算fRMSE的平均值
for var in ['rho_water', 'real_p_t', 'imag_p_t', 'real_Sxx', 'imag_Sxx', 'real_Sxy', 'imag_Sxy', 'real_Syy', 'imag_Syy', 'real_x_u', 'imag_x_u', 'real_x_v', 'imag_x_v']:
    avg_results['fRMSE'][var] = {
        'low': np.mean(all_results['fRMSE'][var]['low']),
        'middle': np.mean(all_results['fRMSE'][var]['middle']),
        'high': np.mean(all_results['fRMSE'][var]['high'])
    }

import logging
from datetime import datetime

log_dir = "VA_result"
os.makedirs(log_dir, exist_ok=True)  # 自动创建目录（如果不存在）
log_file = os.path.join(log_dir, "evaluate_VA.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(message)s',
    filemode='w'  # 'w'覆盖模式，'a'追加模式
)


logging.info(f"参数范围: {offset_range[0]} ~ {offset_range[1]}")

logging.info(f'平均评估结果:')
for metric in avg_results:
    logging.info(f'\n{metric}:')
    if metric != 'fRMSE':
        for var in avg_results[metric]:
            logging.info(f'  {var:10}: {avg_results[metric][var]:.6f}')
    else:
        for var in avg_results[metric]:
            logging.info(f'  {var:10}:')
            for band in avg_results[metric][var]:
                logging.info(f'    {band:7}: {avg_results[metric][var][band]:.6f}')

print(f"评估结果已保存到: {os.path.abspath(log_file)}")