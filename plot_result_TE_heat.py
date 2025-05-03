import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import torch

def evaluate(result, GT_data):
    """
    输入:
        result: dict，预测结果，包含 'mater', 'Ez_real', 'Ez_imag', 'T'
        GT_data: dict，GT数据，包含 'mater_GT', 'real_Ez_GT', 'imag_Ez_GT', 'T_GT'
    输出:
        res_dict: dict，包含 RMSE, nRMSE, MaxError, bRMSE, fRMSE 指标
    """
    res_dict = {
        'RMSE': {}, 'nRMSE': {}, 'MaxError': {}, 'bRMSE': {}, 'fRMSE': {}
    }

    key_map = {
        'mater': ('mater', 'mater_GT'),
        'real_Ez': ('real_Ez', 'real_Ez_GT'),
        'imag_Ez': ('imag_Ez', 'imag_Ez_GT'),
        'T': ('T', 'T_GT'),
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

# 加载数据
TE_heat_results = sio.loadmat("/home/yangchangfan/CODE/DiffusionPDE/TE_heat_result_3k/TE_heat_results_30001.mat")
mater = TE_heat_results['mater']
complex_Ez = TE_heat_results['Ez']
real_Ez = complex_Ez.real
imag_Ez = complex_Ez.imag
T = TE_heat_results['T']

TE_heat_results = {
    'mater': mater,
    'real_Ez': real_Ez,
    'imag_Ez': imag_Ez,
    'T': T
}

mater = np.squeeze(mater)
real_Ez = np.squeeze(real_Ez)
imag_Ez = np.squeeze(imag_Ez)
T = np.squeeze(T)

data_path = '/data/yangchangfan/DiffusionPDE/data/testing/TE_heat'
offset = 30001

# 定义文件路径
mater_GT = sio.loadmat(os.path.join(data_path, 'mater', f'{offset}.mat'))['mater']
complex_Ez_GT = sio.loadmat(os.path.join(data_path, 'Ez', f'{offset}.mat'))['export_Ez']
real_Ez_GT = complex_Ez_GT.real
imag_Ez_GT = complex_Ez_GT.imag
T_GT = sio.loadmat(os.path.join(data_path, 'T', f'{offset}.mat'))['export_T']


vmin_mater = mater_GT.min()
vmax_mater = mater_GT.max()
vmin_real_Ez = real_Ez_GT.min()
vmax_real_Ez = real_Ez_GT.max()
vmin_imag_Ez = imag_Ez_GT.min()
vmax_imag_Ez = imag_Ez_GT.max()
vmin_T = T_GT.min()
vmax_T = T_GT.max()

# 创建 2x4 的子图
fig, axes = plt.subplots(2, 4, figsize=(14, 7))

# 绘制第一行子图
im0 = axes[0, 0].imshow(mater, cmap='plasma', vmin=vmin_mater, vmax=vmax_mater)
axes[0, 0].set_title('mater')
axes[0, 0].axis('off')
plt.colorbar(im0, ax=axes[0, 0])  # 添加 colorbar

im1 = axes[0, 1].imshow(real_Ez, cmap='plasma', vmin=vmin_real_Ez, vmax=vmax_real_Ez)
axes[0, 1].set_title('real_Ez')
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1])  # 添加 colorbar

im2 = axes[0, 2].imshow(imag_Ez, cmap='plasma', vmin=vmin_imag_Ez, vmax=vmax_imag_Ez)
axes[0, 2].set_title('imag_Ez')
axes[0, 2].axis('off')
plt.colorbar(im2, ax=axes[0, 2])  # 添加 colorbar

im3 = axes[0, 3].imshow(T, cmap='plasma', vmin=vmin_T, vmax=vmax_T)
axes[0, 3].set_title('T')
axes[0, 3].axis('off')
plt.colorbar(im3, ax=axes[0, 3])  # 添加 colorbar

# 绘制第二行子图
im4 = axes[1, 0].imshow(mater_GT, cmap='plasma', vmin=vmin_mater, vmax=vmax_mater)
axes[1, 0].set_title('mater_GT')
axes[1, 0].axis('off')
plt.colorbar(im4, ax=axes[1, 0])  # 添加 colorbar

im5 = axes[1, 1].imshow(real_Ez_GT, cmap='plasma', vmin=vmin_real_Ez, vmax=vmax_real_Ez)
axes[1, 1].set_title('real_Ez_GT')
axes[1, 1].axis('off')
plt.colorbar(im5, ax=axes[1, 1])  # 添加 colorbar

im6 = axes[1, 2].imshow(imag_Ez_GT, cmap='plasma', vmin=vmin_imag_Ez, vmax=vmax_imag_Ez)
axes[1, 2].set_title('imag_Ez_GT')
axes[1, 2].axis('off')
plt.colorbar(im6, ax=axes[1, 2])  # 添加 colorbar

im7 = axes[1, 3].imshow(T_GT, cmap='plasma', vmin=vmin_T, vmax=vmax_T)
axes[1, 3].set_title('T_GT')
axes[1, 3].axis('off')
plt.colorbar(im7, ax=axes[1, 3])  # 添加 colorbar

# 调整子图间距
plt.tight_layout()

# 显示图像
plt.show()

# 保存图像
fig.suptitle('TE_heat Results', y=1.02)
plt.savefig('output_TE_heat.png')  # 保存为 PNG 文件


mater = torch.tensor(mater)
mater_GT = torch.tensor(mater_GT)
real_Ez = torch.tensor(real_Ez)
real_Ez_GT = torch.tensor(real_Ez_GT)
imag_Ez = torch.tensor(imag_Ez)
imag_Ez_GT = torch.tensor(imag_Ez_GT)
T = torch.tensor(T)
T_GT = torch.tensor(T_GT)

# 计算相对误差
relative_error_mater = torch.norm(mater - mater_GT, 2) / torch.norm(mater_GT, 2)
relative_error_real_Ez = torch.norm(real_Ez - real_Ez_GT, 2) / torch.norm(real_Ez_GT, 2)
relative_error_imag_Ez = torch.norm(imag_Ez - imag_Ez_GT, 2) / torch.norm(imag_Ez_GT, 2)
relative_error_T = torch.norm(T - T_GT, 2) / torch.norm(T_GT, 2)

print(f'Relative error of mater: {relative_error_mater}')
print(f'Relative error of real_Ez: {relative_error_real_Ez}')
print(f'Relative error of imag_Ez: {relative_error_imag_Ez}')
print(f'Relative error of T: {relative_error_T}')




TE_heat_GT = {
    'mater_GT': mater_GT,
    'real_Ez_GT': real_Ez_GT,
    'imag_Ez_GT': imag_Ez_GT,
    'T_GT': T_GT
}

res_dict = evaluate(TE_heat_results, TE_heat_GT)

# 打印结果
print('-' * 20)
print('Evaluation metrics:')
for metric in res_dict:
    for var in res_dict[metric]:
        print(f'{metric:8} {var:10}: {res_dict[metric][var]}')