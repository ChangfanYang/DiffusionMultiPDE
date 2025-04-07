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
    

def generate_separa_mater(mater, T, poly_GT, mater_iden, device=torch.device('cuda')):
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


def get_NS_heat_loss(Q_heat, u_u, u_v, T, Q_heat_GT, u_u_GT, u_v_GT, T_GT, Q_heat_mask, u_u_mask, u_v_mask, T_mask, device=torch.device('cuda')):
    """Return the loss of the NS_heat equation and the observation loss."""
    # sigma, pho, K_E = generate_separa_mater_NS(mater, T, poly_GT, mater_iden)

    # delta_x = 1e-3 # 1mm
    # delta_y = 1e-3 # 1mm
    
    # deriv_x = torch.tensor([[1, 0, -1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * delta_x)
    # deriv_y = torch.tensor([[1], [0], [-1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * delta_y)

    # deriv_x_complex = torch.complex(deriv_x, torch.zeros_like(deriv_x))
    # deriv_y_complex = torch.complex(deriv_y, torch.zeros_like(deriv_y))

    # # E_filed
    # grad_x_next_x_E = F.conv2d(Ez, deriv_x_complex, padding=(0, 1))
    # grad_x_next_y_E = F.conv2d(Ez, deriv_y_complex, padding=(1, 0))
    # Laplac_E = F.conv2d(grad_x_next_x_E, deriv_x_complex, padding=(0, 1)) + F.conv2d(grad_x_next_y_E, deriv_y_complex, padding=(1, 0))
    # result_E = Laplac_E + K_E * Ez
    # # T_filed
    # grad_x_next_x_T = F.conv2d(T, deriv_x, padding=(0, 1))
    # grad_x_next_y_T = F.conv2d(T, deriv_y, padding=(1, 0))
    # Laplac_T = F.conv2d(grad_x_next_x_T, deriv_x, padding=(0, 1)) + F.conv2d(grad_x_next_y_T, deriv_y, padding=(1, 0))
    # result_T = pho * Laplac_T + 0.5 * sigma * Ez * torch.conj(Ez)

    pde_loss_NS = 0
    pde_loss_heat = 0

    # pde_loss_NS = pde_loss_NS.squeeze()
    # pde_loss_heat = pde_loss_heat.squeeze()
    
    # scipy.io.savemat('pde_loss_NS.mat', {'pde_loss_NS': pde_loss_NS.cpu().detach().numpy()})
    # scipy.io.savemat('pde_loss_heat.mat', {'pde_loss_heat': pde_loss_heat.cpu().detach().numpy()})
    # scipy.io.savemat('result_E.mat', {'result_E': result_E.cpu().detach().numpy()})
    # scipy.io.savemat('Laplac_E.mat', {'Laplac_E': Laplac_E.cpu().detach().numpy()})
    # scipy.io.savemat('result_T.mat', {'result_T': result_T.cpu().detach().numpy()})


    observation_loss_Q_heat = (Q_heat - Q_heat_GT).squeeze()
    observation_loss_Q_heat = observation_loss_Q_heat * Q_heat_mask  
    observation_loss_u_u = (u_u - u_u_GT).squeeze()
    observation_loss_u_u = observation_loss_u_u * u_u_mask  
    observation_loss_u_v = (u_v - u_v_GT).squeeze()
    observation_loss_u_v = observation_loss_u_v * u_v_mask  
    observation_loss_T = (T - T_GT).squeeze()
    observation_loss_T = observation_loss_T * T_mask

    return pde_loss_NS, pde_loss_heat, observation_loss_Q_heat, observation_loss_u_u, observation_loss_u_v, observation_loss_T

def generate_NS_heat(config):
    """Generate NS_heat equation."""
    ############################ Load data and network ############################
    datapath = config['data']['datapath']
    offset = config['data']['offset'][0]
    device = config['generate']['device']

    Q_heat_GT_path = os.path.join(datapath, "Q_heat", f"{offset}.mat")
    # print(Q_heat_GT_path)

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
    
    batch_size = config['generate']['batch_size']
    seed = config['generate']['seed']
    torch.manual_seed(seed)
    
    network_pkl = config['test']['pre-trained']
    print(f'Loading networks from "{network_pkl}"...')
    f = open(network_pkl, 'rb')
    net = pickle.load(f)['ema'].to(device)
    
    ############################ Set up EDM latent ############################
    print(f'Generating {batch_size} samples...')
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]
    
    sigma_min = config['generate']['sigma_min']
    sigma_max = config['generate']['sigma_max']
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    
    num_steps = config['test']['iterations']
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    rho = config['generate']['rho']
    sigma_t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigma_t_steps = torch.cat([net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])]) # t_N = 0
    
    x_next = latents.to(torch.float64) * sigma_t_steps[0]
    known_index_Q_heat = random_index(500, 128, seed=3)
    known_index_u_u = random_index(500, 128, seed=2)
    known_index_u_v = random_index(500, 128, seed=1)
    known_index_T = random_index(500, 128, seed=0)
    
    ############################ Sample the data ############################
    for i, (sigma_t_cur, sigma_t_next) in tqdm.tqdm(list(enumerate(zip(sigma_t_steps[:-1], sigma_t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next.detach().clone()
        x_cur.requires_grad = True
        sigma_t = net.round_sigma(sigma_t_cur)
        
        # Euler step
        x_N = net(x_cur, sigma_t, class_labels=class_labels).to(torch.float64)
        d_cur = (x_cur - x_N) / sigma_t
        x_next = x_cur + (sigma_t_next - sigma_t) * d_cur
        
        # 2nd order correction
        if i < num_steps - 1:
            x_N = net(x_next, sigma_t_next, class_labels=class_labels).to(torch.float64)
            d_prime = (x_next - x_N) / sigma_t_next
            x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)
        
        # Scale the data back
        Q_heat_N = x_N[:,0,:,:].unsqueeze(0)
        u_u_N = x_N[:,1,:,:].unsqueeze(0)
        u_v_N = x_N[:,2,:,:].unsqueeze(0)
        T_N = x_N[:,3,:,:].unsqueeze(0)

        # inv_normalization
        range_allQ_heat_paths = "/data/yangchangfan/DiffusionPDE/data/training/NS_heat/Q_heat/range_allQ_heat.mat"
        range_allQ_heat = sio.loadmat(range_allQ_heat_paths)['range_allQ_heat']
        range_allQ_heat = torch.tensor(range_allQ_heat, device=device)

        max_Q_heat = range_allQ_heat[0,1]
        min_Q_heat = range_allQ_heat[0,0]

        range_allu_u_paths = "/data/yangchangfan/DiffusionPDE/data/training/NS_heat/u_u/range_allu_u.mat"
        range_allu_u = sio.loadmat(range_allu_u_paths)['range_allu_u']
        range_allu_u = torch.tensor(range_allu_u, device=device)

        max_u_u = range_allu_u[0,1]
        min_u_u = range_allu_u[0,0]

        range_allu_v_paths = "/data/yangchangfan/DiffusionPDE/data/training/NS_heat/u_v/range_allu_v.mat"
        range_allu_v = sio.loadmat(range_allu_v_paths)['range_allu_v']
        range_allu_v = torch.tensor(range_allu_v, device=device)

        max_u_v = range_allu_v[0,1]
        min_u_v = range_allu_v[0,0]

        range_allT_paths = "/data/yangchangfan/DiffusionPDE/data/training/NS_heat/T/range_allT.mat"
        range_allT = sio.loadmat(range_allT_paths)['range_allT']
        range_allT = torch.tensor(range_allT, device=device)

        max_T = range_allT[0,1]
        min_T = range_allT[0,0]

        Q_heat_N = ((Q_heat_N+0.9)/1.8 *(max_Q_heat - min_Q_heat) + min_Q_heat).to(torch.float64)
        u_u_N = ((u_u_N+0.9)/1.8 *(max_u_u - min_u_u) + min_u_u).to(torch.float64)
        u_v_N = ((u_v_N+0.9)/1.8 *(max_u_v - min_u_v) + min_u_v).to(torch.float64)
        T_N = ((T_N+0.9)/1.8 *(max_T - min_T) + min_T).to(torch.float64)


        # Compute the loss

        pde_loss_NS, pde_loss_heat, observation_loss_Q_heat, observation_loss_u_u, observation_loss_u_v, observation_loss_T = get_NS_heat_loss(Q_heat_N, u_u_N, u_v_N, T_N, Q_heat_GT, u_u_GT, u_v_GT, T_GT, known_index_Q_heat, known_index_u_u, known_index_u_v, known_index_T, device=device)
        
        # L_pde_NS = torch.norm(pde_loss_NS, 2)/(128*128)
        # L_pde_heat = torch.norm(pde_loss_heat, 2)/(128*128)
        

        L_obs_Q_heat = torch.norm(observation_loss_Q_heat, 2)/500
        L_obs_u_u = torch.norm(observation_loss_u_u, 2)/500
        L_obs_u_v = torch.norm(observation_loss_u_v, 2)/500
        L_obs_T = torch.norm(observation_loss_T, 2)/500

        # print(L_pde)
        # print(L_obs_mater)
        # print(L_obs_Ez)
        # print(L_obs_T)

        output_file_path = "inference_losses.jsonl"
        if i % 5 == 0:
            log_entry = {
              "step": i,
            #   "L_pde_NS": L_pde_NS.tolist(),
            #   "L_pde_heat": L_pde_heat.tolist(),
               "L_obs_Q_heat": L_obs_Q_heat.tolist(),
               "L_obs_u_u": L_obs_u_u.tolist(),
               "L_obs_u_v": L_obs_u_v.tolist(),
              "L_obs_T": L_obs_T.tolist()
           }
            with open(output_file_path, "a") as file:
                json.dump(log_entry, file)
                file.write("\n")  

        grad_x_cur_obs_Q_heat = torch.autograd.grad(outputs=L_obs_Q_heat, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_obs_u_u = torch.autograd.grad(outputs=L_obs_u_u, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_obs_u_v = torch.autograd.grad(outputs=L_obs_u_v, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_obs_T = torch.autograd.grad(outputs=L_obs_T, inputs=x_cur, retain_graph=True)[0]
        # grad_x_cur_pde_NS = torch.autograd.grad(outputs=L_pde_NS, inputs=x_cur, retain_graph=True)[0]
        # grad_x_cur_pde_heat = torch.autograd.grad(outputs=L_pde_heat, inputs=x_cur)[0]

        # zeta_obs_mater = config['generate']['zeta_obs_mater']
        # zeta_obs_Ez = config['generate']['zeta_obs_Ez']
        # zeta_obs_T = 1e3*config['generate']['zeta_obs_T']
        # zeta_pde = config['generate']['zeta_pde']
       
        zeta_obs_Q_heat = 10
        zeta_obs_u_u = 10
        zeta_obs_u_v = 10
        zeta_obs_T = 10
        # zeta_pde_NS = 10
        # zeta_pde_heat = 10

    # scale zeta
        norm_Q_heat = torch.norm(zeta_obs_Q_heat * grad_x_cur_obs_Q_heat)
        scale_factor = 1.0 / norm_Q_heat
        zeta_obs_Q_heat = zeta_obs_Q_heat * scale_factor

        norm_u_u = torch.norm(zeta_obs_u_u * grad_x_cur_obs_u_u)
        scale_factor = 1.0 / norm_u_u
        zeta_obs_u_u = zeta_obs_u_u * scale_factor

        norm_u_v = torch.norm(zeta_obs_u_v * grad_x_cur_obs_u_v)
        scale_factor = 1.0 / norm_u_v
        zeta_obs_u_v = zeta_obs_u_v * scale_factor

        norm_T = torch.norm(zeta_obs_T * grad_x_cur_obs_T)
        scale_factor = 1.0 / norm_T
        zeta_obs_T = zeta_obs_T * scale_factor
        

        if i <= 1 * num_steps:
            x_next = x_next - zeta_obs_Q_heat * grad_x_cur_obs_Q_heat - zeta_obs_u_u * grad_x_cur_obs_u_u - zeta_obs_u_v * grad_x_cur_obs_u_v - zeta_obs_T * grad_x_cur_obs_T
      
            # norm_value = torch.norm(zeta_obs_Q_heat * grad_x_cur_obs_Q_heat).item()
            # print(norm_value)

            # x_next = x_next

        else:
            
            norm_pde_NS = torch.norm(zeta_pde_NS * grad_x_cur_pde_NS)
            scale_factor = 1 / norm_pde_NS
            zeta_pde_NS = zeta_pde_NS * scale_factor

            norm_pde_heat = torch.norm(zeta_pde_heat * grad_x_cur_pde_heat)
            scale_factor = 1 / norm_pde_heat
            zeta_pde_heat = zeta_pde_heat * scale_factor

            # x_next = x_next - 0.1 * (zeta_obs_mater * grad_x_cur_obs_mater + zeta_obs_Ez * grad_x_cur_obs_Ez + zeta_obs_T * grad_x_cur_obs_T) - zeta_pde_E * grad_x_cur_pde_E - zeta_pde_T * grad_x_cur_pde_T

            x_next = x_next - 0.8*(zeta_obs_Q_heat * grad_x_cur_obs_Q_heat + zeta_obs_u_u * grad_x_cur_obs_u_u + zeta_obs_u_v * grad_x_cur_obs_u_v + zeta_obs_T * grad_x_cur_obs_T) - 0.2* (zeta_pde_NS * grad_x_cur_pde_NS + zeta_pde_heat * grad_x_cur_pde_heat)


            # norm_value = torch.norm(zeta_pde_NS * grad_x_cur_pde_NS).item()
            # print(norm_value)

    ############################ Save the data ############################
    x_final = x_next
    Q_heat_final = x_final[:,0,:,:].unsqueeze(0)
    u_u_final = x_final[:,1,:,:].unsqueeze(0)
    u_v_final = x_final[:,2,:,:].unsqueeze(0)
    T_final = x_final[:,3,:,:].unsqueeze(0)    
    

    Q_heat_final = ((Q_heat_final+0.9)/1.8 *(max_Q_heat - min_Q_heat) + min_Q_heat).to(torch.float64)
    u_u_final = ((u_u_final+0.9)/1.8 *(max_u_u - min_u_u) + min_u_u).to(torch.float64)
    u_v_final = ((u_v_final+0.9)/1.8 *(max_u_v - min_u_v) + min_u_v).to(torch.float64)
    T_final = ((T_final+0.9)/1.8 *(max_T - min_T) + min_T).to(torch.float64)


    relative_error_Q_heat = torch.norm(Q_heat_final - Q_heat_GT, 2) / torch.norm(Q_heat_GT, 2)
    relative_error_u_u = torch.norm(u_u_final - u_u_GT, 2) / torch.norm(u_u_GT, 2)
    relative_error_u_v = torch.norm(u_v_final - u_v_GT, 2) / torch.norm(u_v_GT, 2)
    relative_error_T = torch.norm(T_final - T_GT, 2) / torch.norm(T_GT, 2)  
    
    # scipy.io.savemat('T_final.mat', {'T_final': T_final.cpu().detach().numpy()})

    # print(u_u_final)
    # exit()

    print(f'Relative error of Q_heat: {relative_error_Q_heat}')
    print(f'Relative error of u_u: {relative_error_u_u}')
    print(f'Relative error of u_v: {relative_error_u_v}')
    print(f'Relative error of T: {relative_error_T}')

    Q_heat_final = Q_heat_final.detach().cpu().numpy()
    u_u_final = u_u_final.detach().cpu().numpy()
    u_v_final = u_v_final.detach().cpu().numpy()
    T_final = T_final.detach().cpu().numpy()

    scipy.io.savemat('NS_heat_results.mat', {'Q_heat': Q_heat_final, 'u_u': u_u_final, 'u_v': u_v_final, 'T': T_final})
    print('Done.')