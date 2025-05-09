import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import torch

def evaluate(result, GT_data):
    """
    输入:
        result: dict，预测结果，包含 'kappa', 'ec_V', 'u_flow', 'v_flow'
        GT_data: dict，GT数据，包含 'kappa', 'ec_V', 'u_flow', 'v_flow'
    输出:
        res_dict: dict，包含 RMSE, nRMSE, MaxError, bRMSE, fRMSE 指标
    """
    res_dict = {
        'RMSE': {}, 'nRMSE': {}, 'MaxError': {}, 'bRMSE': {}, 'fRMSE': {}
    }

    key_map = {
        'kappa': ('kappa', 'kappa_GT'),
        'ec_V': ('ec_V', 'ec_V_GT'),
        'u_flow': ('u_flow', 'u_flow_GT'),
        'v_flow': ('v_flow', 'v_flow_GT'),
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
        test1 = (pred - gt) ** 2
        test2 = torch.mean((pred - gt) ** 2)
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
data_path = '/data/yangchangfan/DiffusionPDE/data/testing/E_flow'
results_path = '/home/yangchangfan/CODE/DiffusionPDE/E_flow_result'

# 初始化存储所有结果的字典
all_results = {
    'RMSE': {'kappa': [], 'ec_V': [], 'u_flow': [], 'v_flow': []},
    'nRMSE': {'kappa': [], 'ec_V': [], 'u_flow': [], 'v_flow': []},
    'MaxError': {'kappa': [], 'ec_V': [], 'u_flow': [], 'v_flow': []},
    'bRMSE': {'kappa': [], 'ec_V': [], 'u_flow': [], 'v_flow': []},
    'fRMSE': {'kappa': {'low': [], 'middle': [], 'high': []}, 
              'ec_V': {'low': [], 'middle': [], 'high': []},
              'u_flow': {'low': [], 'middle': [], 'high': []},
              'v_flow': {'low': [], 'middle': [], 'high': []}}
}

offset_range=[10001, 11001]
for idx in range(offset_range[0], offset_range[1]):
    try:
        # 加载预测结果
        pred = sio.loadmat(f'{results_path}/E_flow_results_{idx}.mat')
        kappa = pred['kappa']
        ec_V = pred['ec_V']
        u_flow = pred['u_flow']
        v_flow = pred['v_flow']
        
        E_flow_results = {
            'kappa': kappa,
            'ec_V': ec_V,
            'u_flow': u_flow,
            'v_flow': v_flow,
        }

        # 加载GT数据
        E_flow_GT = {
            'kappa_GT': sio.loadmat(f'{data_path}/kappa/{idx}.mat')['export_kappa'],
            'ec_V_GT': sio.loadmat(f'{data_path}/ec_V/{idx}.mat')['export_ec_V'],    
            'u_flow_GT': sio.loadmat(f'{data_path}/u_flow/{idx}.mat')['export_u_flow'],  
            'v_flow_GT': sio.loadmat(f'{data_path}/v_flow/{idx}.mat')['export_v_flow']
        }
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        # print(NS_heat_GT['Q_heat_GT'])
        # print(NS_heat_results['Q_heat'])
        # exit()
        # 评估当前数据
        res_dict = evaluate(E_flow_results, E_flow_GT)
        
        # 保存所有结果
        for metric in ['RMSE', 'nRMSE', 'MaxError', 'bRMSE']:
            for var in ['kappa', 'ec_V', 'u_flow', 'v_flow']:
                all_results[metric][var].append(res_dict[metric][var])
        
        for var in ['kappa', 'ec_V', 'u_flow', 'v_flow']:
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
    for var in ['kappa', 'ec_V', 'u_flow', 'v_flow']:
        avg_results[metric][var] = np.mean(all_results[metric][var])

# 计算fRMSE的平均值
for var in ['kappa', 'ec_V', 'u_flow', 'v_flow']:
    avg_results['fRMSE'][var] = {
        'low': np.mean(all_results['fRMSE'][var]['low']),
        'middle': np.mean(all_results['fRMSE'][var]['middle']),
        'high': np.mean(all_results['fRMSE'][var]['high'])
    }

import logging
from datetime import datetime

log_dir = "E_flow_result"
os.makedirs(log_dir, exist_ok=True)  # 自动创建目录（如果不存在）
log_file = os.path.join(log_dir, "evaluate_E_flow.log")

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