import os
import numpy as np
import torch
import scipy.io as sio
from scipy.io import loadmat

import logging
from datetime import datetime

import os
import numpy as np
import torch
import scipy.io as sio
from scipy.io import loadmat

res_dict = {
    'RMSE': {}, 'nRMSE': {}, 'MaxError': {}, 'bRMSE': {}, 'fRMSE_low': {}, 'fRMSE_mid': {}, 'fRMSE_high': {},
}

for metric in res_dict:
    for key in ['S_c']:
        res_dict[metric][key] = 0

    for key in ['u_u', 'u_v', 'c_flow']:
        res_dict[metric][key] = {}
        for t in range(11):
            res_dict[metric][key][f'{t}'] = 0

def evaluate(Elder_result, Elder_GT):
    # 变量映射及索引范围（t1~t10）
    for var in Elder_GT.keys():
        pred = torch.tensor(Elder_result[var]).squeeze()
        gt = torch.tensor(Elder_GT[var].squeeze())

        if var == 'S_c':
            diff = pred - gt
            # RMSE
            rmse = torch.sqrt(torch.mean(diff ** 2)).item()
            res_dict['RMSE'][var] += rmse

            # nRMSE
            nrmse = torch.norm(diff) / torch.norm(gt)
            res_dict['nRMSE'][var] += nrmse.item()

            #
            maxerr = torch.max(torch.abs(diff)).item()
            res_dict['MaxError'][var] += maxerr

            # bRMSE（边界）
            boundary_mask = torch.zeros_like(gt, dtype=torch.bool)
            boundary_mask[0, :] = boundary_mask[-1, :] = True
            boundary_mask[:, 0] = boundary_mask[:, -1] = True
            diff_b = diff[boundary_mask]
            brmse = torch.sqrt(torch.mean(diff_b ** 2)).item()
            res_dict['bRMSE'][var] += brmse

            # fRMSE（FFT误差）
            fRMSE_var = {}
            freq_bands = {'low': (0, 4), 'middle': (5, 12), 'high': (13, None)}
            pred_fft = torch.fft.fft2(pred)
            gt_fft = torch.fft.fft2(gt)
            H, W = pred_fft.shape
            kx = torch.fft.fftfreq(H).reshape(-1, 1).expand(H, W)
            ky = torch.fft.fftfreq(W).reshape(1, -1).expand(H, W)
            radius = torch.sqrt(kx ** 2 + ky ** 2) * max(H, W)
            for band, (k_min, k_max) in freq_bands.items():
                if k_max is None:
                    mask = (radius >= k_min)
                else:
                    mask = (radius >= k_min) & (radius <= k_max)
                diff_fft = torch.abs(pred_fft - gt_fft) ** 2
                band_error = diff_fft[mask].mean().sqrt().item()

                if band not in fRMSE_var:
                    fRMSE_var[band] = []
                fRMSE_var[band].append(band_error)
            res_dict[f'fRMSE_low'][var] += np.mean(fRMSE_var['low'])
            res_dict[f'fRMSE_mid'][var] += np.mean(fRMSE_var['middle'])
            res_dict[f'fRMSE_high'][var] += np.mean(fRMSE_var['high'])
        else:
            for t in range(pred.shape[0]):
                diff = pred[t] - gt[t]

                # RMSE
                rmse = torch.sqrt(torch.mean(diff ** 2)).item()
                res_dict['RMSE'][var][f'{t}'] += rmse

                # nRMSE
                nrmse = torch.norm(diff) / torch.norm(gt[t])
                res_dict['nRMSE'][var][f'{t}'] += nrmse.item()

                #
                maxerr = torch.max(torch.abs(diff)).item()
                res_dict['MaxError'][var][f'{t}'] += maxerr

                # bRMSE（边界）
                boundary_mask = torch.zeros_like(pred[t], dtype=torch.bool)
                boundary_mask[0, :] = boundary_mask[-1, :] = True
                boundary_mask[:, 0] = boundary_mask[:, -1] = True
                diff_b = diff[boundary_mask]
                brmse = torch.sqrt(torch.mean(diff_b ** 2)).item()
                res_dict['bRMSE'][var][f'{t}'] += brmse

                # fRMSE（FFT误差）
                fRMSE_var = {}
                freq_bands = {'low': (0, 4), 'middle': (5, 12), 'high': (13, None)}
                pred_fft = torch.fft.fft2(pred[t])
                gt_fft = torch.fft.fft2(gt[t])
                H, W = pred_fft.shape
                kx = torch.fft.fftfreq(H).reshape(-1, 1).expand(H, W)
                ky = torch.fft.fftfreq(W).reshape(1, -1).expand(H, W)
                radius = torch.sqrt(kx ** 2 + ky ** 2) * max(H, W)
                for band, (k_min, k_max) in freq_bands.items():
                    if k_max is None:
                        mask = (radius >= k_min)
                    else:
                        mask = (radius >= k_min) & (radius <= k_max)
                    diff_fft = torch.abs(pred_fft - gt_fft) ** 2
                    band_error = diff_fft[mask].mean().sqrt().item()

                    if band not in fRMSE_var:
                        fRMSE_var[band] = []
                    fRMSE_var[band].append(band_error)
                res_dict[f'fRMSE_low'][var][f'{t}'] += np.mean(fRMSE_var['low'])
                res_dict[f'fRMSE_mid'][var][f'{t}'] += np.mean(fRMSE_var['middle'])
                res_dict[f'fRMSE_high'][var][f'{t}'] += np.mean(fRMSE_var['high'])
    return res_dict


def load_gt_data(data_test_path, offset, device):
    data_gt = {'S_c': None, 'u_u': None, 'u_v': None, 'c_flow': None}

    # S_c
    path_Sc = os.path.join(data_test_path, 'S_c', str(offset), '0.mat')
    Sc_data = loadmat(path_Sc)
    Sc = Sc_data['export_S_c']
    data_gt['S_c'] = Sc

    # u_u, u_v, c_flow
    var_names = ['u_u', 'u_v', 'c_flow']
    for var_idx, var in enumerate(var_names):
        time_steps = 11
        cur_data = np.zeros((time_steps, 128, 128), dtype=np.float64)
        for t in range(time_steps):
            path_t = os.path.join(data_test_path, var, str(offset), f'{t}.mat')
            data_t = loadmat(path_t)[f'export_{var}']
            cur_data[t, :, :] = data_t
        data_gt[var] = cur_data
    return data_gt



def log_nested_dict(data, indent=0):
    for key, value in data.items():
        if isinstance(value, dict):
            logging.info('  ' * indent + f'{key}:')
            log_nested_dict(value, indent + 1)
        elif isinstance(value, (int, float)):
            # 科学计数法显示极小值，普通浮点数显示6位小数
            if abs(value) < 1e-5:
                logging.info('  ' * indent + f'{key:10}: {value:.6e}')
            else:
                logging.info('  ' * indent + f'{key:10}: {value:.6f}')
        else:
            logging.info('  ' * indent + f'{key:10}: {value}')

            
if __name__ == "__main__":
    # 路径配置
    data_test_path = '/data/yangchangfan/DiffusionPDE/data/testing/Elder'
    results_path = '/home/yangchangfan/CODE/DiffusionPDE/Elder_result'

    offset_range = [1001, 1101]
    print(f"开始评估区间: {offset_range[0]} ~ {offset_range[1]}")

    all_results = {
        metric: {'ec_V': [], 'u_u': [], 'u_v': []}
        for metric in ['RMSE', 'nRMSE', 'MaxError', 'bRMSE']
    }

    all_results['fRMSE'] = {
        var: {'low': [], 'middle': [], 'high': []}
        for var in ['ec_V', 'u_u', 'u_v']
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for offset in range(offset_range[0], offset_range[1]):
        pred_data = sio.loadmat(f'{results_path}/Elder_results_{offset}.mat')
        gt_data = load_gt_data(data_test_path, offset, device)
        evaluate(pred_data, gt_data)

    for metric in res_dict:
        for key in ['S_c']:
            res_dict[metric][key] /= (offset_range[1]-offset_range[0])

        for key in ['u_u', 'u_v', 'c_flow']:
            for t in range(11):
                res_dict[metric][key][f'{t}'] /= (offset_range[1]-offset_range[0])

    # print(res_dict)

    # 初始化日志
    log_dir = "Elder_result"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "evaluate_Elder.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(message)s',
        filemode='w'
    )

    # 记录当前offset的结果
    logging.info(f"\n=== Offset {offset} ===")
    for metric in res_dict:
        logging.info(f"{metric}:")
        log_nested_dict(res_dict[metric], indent=1)

    print(f"评估结果已保存到: {os.path.abspath(log_file)}")


