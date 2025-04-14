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
import json
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datapath = '/data/yangchangfan/DiffusionPDE/data/testing/TE_heat'
offset = 3

mater_GT_path = os.path.join(datapath, "mater", f"{offset}.mat")
mater_GT = sio.loadmat(mater_GT_path)['mater']
mater_GT = torch.tensor(mater_GT, dtype=torch.float64, device=device)

Ez_GT_path = os.path.join(datapath, "Ez", f"{offset}.mat")
Ez_GT = sio.loadmat(Ez_GT_path)['export_Ez']
Ez_GT = torch.tensor(Ez_GT, dtype=torch.complex128, device=device)
    
T_GT_path = os.path.join(datapath, "T", f"{offset}.mat")
T_GT = sio.loadmat(T_GT_path)['export_T']
T_GT = torch.tensor(T_GT, dtype=torch.float64, device=device)

poly_GT_path = os.path.join(datapath, "polycsv", f"{offset}.csv")
poly_GT = pd.read_csv(poly_GT_path, header=None)
poly_GT = torch.tensor(poly_GT.values, dtype=torch.float64)

mater_final = sio.loadmat('/home/yangchangfan/CODE/DiffusionPDE/TE_heat_results.mat')['mater']
complex_Ez_final = sio.loadmat('/home/yangchangfan/CODE/DiffusionPDE/TE_heat_results.mat')['Ez']
T_final = sio.loadmat('/home/yangchangfan/CODE/DiffusionPDE/TE_heat_results.mat')['T']


mater_final = torch.from_numpy(mater_final.squeeze()).to(mater_GT.device)
complex_Ez_final =  torch.from_numpy(complex_Ez_final.squeeze()).to(mater_GT.device)
T_final = torch.from_numpy(T_final.squeeze()).to(mater_GT.device)

relative_error_mater = torch.norm(mater_final - mater_GT, 2) / torch.norm(mater_GT, 2)
relative_error_Ez = torch.norm(complex_Ez_final - Ez_GT, 2) / torch.norm(Ez_GT, 2)
relative_error_T = torch.norm(T_final - T_GT, 2) / torch.norm(T_GT, 2)  
    
scipy.io.savemat('T_final.mat', {'T_final': T_final.cpu().detach().numpy()})

print(f'Relative error of mater: {relative_error_mater}')
print(f'Relative error of Ez: {relative_error_Ez}')
print(f'Relative error of T: {relative_error_T}')