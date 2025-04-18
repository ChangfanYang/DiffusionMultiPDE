
import numpy as np
import scipy.io
import os
import scipy.io as sio


# index = 1
index = 10001 - 1  # 注意你offset是10001，而npy是merge_1.npy对应原始1.mat

npy_data = np.load(f"/data/yangchangfan/DiffusionPDE/data/VA-merged/merge_{index+1}.npy")
rho_water_from_npy = npy_data[:, :, 0]  # 第一个通道

rho_water_GT_mat = sio.loadmat(f"/data/yangchangfan/DiffusionPDE/data/testing/VA/rho_water/{index+1}.mat")['export_rho_water']
range_rho = sio.loadmat(f"/data/yangchangfan/DiffusionPDE/data/training/VA/rho_water/range_allrho_water.mat")['range_allrho_water']
min_val, max_val = range_rho[0,0], range_rho[0,1]

# 用 GT 值归一化后再对比
rho_water_GT_norm = (rho_water_GT_mat - min_val) / (max_val - min_val + 1e-10) * 1.8 - 0.9

print("误差均值:", np.mean(np.abs(rho_water_from_npy - rho_water_GT_norm)))
print("误差最大:", np.max(np.abs(rho_water_from_npy - rho_water_GT_norm)))
