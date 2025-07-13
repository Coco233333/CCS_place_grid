import numpy as np
import matplotlib.pyplot as plt
from Network_bayesian import Place_net, Grid_net, Coupled_Net
import brainpy as bp
import brainpy.math as bm
from matplotlib.animation import FuncAnimation
import time
from scipy.stats import ttest_ind, norm

# 圆周距离函数
def circ_dis(phi_1, phi_2):
    dis = phi_1 - phi_2
    dis = np.where(dis > np.pi, dis - 2 * np.pi, dis)
    dis = np.where(dis < -np.pi, dis + 2 * np.pi, dis)
    return dis

# 默认参数
# grid spacing
lambda_1 = 3
lambda_2 = 4
lambda_3 = 5
Lambda = np.array([lambda_1, lambda_2, lambda_3])
L = lambda_1 * lambda_2 * lambda_3
# cell number
num_p = int(200)
rho_p = num_p / L
rho_g = rho_p
num_g = int(rho_g * 2 * np.pi)  # 为了让两个网络的rho相等
M = len(Lambda)
# feature space
x = np.linspace(0, L, num_p, endpoint=False)
theta = np.linspace(0, 2 * np.pi, num_g, endpoint=False)
# connection range
a_p = 0.3
a_g = a_p / Lambda * 2 * np.pi
# connection strength
J_p = 20
J_g = J_p
J_pg = J_p / 25


# divisive normalization
k_p = 20.
k_g = Lambda / 2 / np.pi * k_p
# time constants
tau_p = 1
tau_g = 2 * np.pi * tau_p / Lambda
# input_strength
alpha_p = 0.05
alpha_g = 0.05

noise_ratio = 0.007
Ap = 1.0084058
Rp = 0.0128615275
Ag = 0.9814125
Rg = 0.013212965

def Grid_tuning_generation(phi_candidate, a_g, num_g):
    # phi_candidate shape: [M, n_candidate]
    n_candidate = phi_candidate.shape[-1]
    theta = np.linspace(0, 2 * np.pi, num_g, endpoint=False)  # Assuming theta is linspace
    
    # Broadcasting phi_candidate to shape [M, 1, n_candidate]
    phi_candidate_expanded = phi_candidate[:, np.newaxis, :]
    
    # Calculate dis_theta using broadcasting
    dis_theta = circ_dis(theta[:, np.newaxis], phi_candidate_expanded)
    
    # Calculate fg_prime using broadcasting
    a_g_expanded = a_g[:, np.newaxis, np.newaxis]
    fg_prime = np.exp(-dis_theta ** 2 / (4 * a_g_expanded ** 2))
    
    return fg_prime



def loglikelihood_Ig(Ig, fg, sigma_g):
    n_phi = fg.shape[-1]
    # 使用 expand_dims 将矩阵扩展为 (n, m, 1)
    Ig_expand = np.expand_dims(Ig, axis=-1)
    # 使用 tile 将矩阵沿最后一个轴重复 K 次
    Ig_expand = np.tile(Ig_expand, (1, 1, n_phi))
    log_prob = -0.5 * (Ig_expand - fg)**2 / sigma_g[:,None,None]**2 # shape [M, n_g, n_phi]
    log_prob = np.sum(log_prob, axis=1) 
    return log_prob # shape [M, n_phi]

def mapping_func(x):
    lambda_gs = Lambda
    phi = x[:,None] % Lambda *np.pi*2/Lambda
    return phi # [n_pos, M]

def prior_function(phi, z_candidates, sigma_phi):
    '''
    P(phi | x), phi shape [M], phi_x is a scalar
    '''
    phi_x = mapping_func(z_candidates) # shape [n_pos, M]
    kappa_phi = 1 / (sigma_phi)**2
    log_prob = kappa_phi*np.cos(phi-phi_x) # shape [n_pos, M]
    log_prob_z = np.sum(log_prob, axis=1)
    return log_prob_z # shape [M]

def Simplified_PSC_MAP_decoder(activation_gs, n_pos=10000, n_phi=500, M=3, 
                               alpha_p_infer=0.05, alpha_g_infer=0.05, Ap=Ap, Rp=Rp, Ag=Ag):  
    '''
    MAP: Maximum A Posteriori
    activation_gs shape [M, n_g]
    '''
    sigma_g = np.sqrt(np.sqrt(np.pi) * Ag ** 3 * rho_g * tau_g / (a_g * alpha_g_infer))
    sigma_phi = np.sqrt(8 * np.pi * Ag * tau_g / (Lambda * J_pg * rho_p * Rp))
    sigma_g_infer = sigma_g * noise_ratio
    sigma_phi_infer = sigma_phi * noise_ratio
    
    L_env = 60
    ## parameter space
    z_candidates = np.linspace(0, L_env, n_pos)
    
    phi_candidates = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    phi_candidates_modules = np.tile(phi_candidates[:, np.newaxis], (1, M))
    phi_candidates_modules = phi_candidates_modules.T
    fg_modules = Grid_tuning_generation(phi_candidates_modules,a_g,num_g) # shape [M, n_g, n_phi]

    log_likelihood_fr = loglikelihood_Ig(activation_gs, fg_modules, sigma_g=sigma_g_infer) # shape [M, n_phi]
    phi_decode_index = np.argmax(log_likelihood_fr, axis=1)

    phi_decode = phi_candidates[phi_decode_index]
    # Second step: decode z
    prior = prior_function(phi_decode, z_candidates, sigma_phi=sigma_phi_infer)
    # plt.plot(prior)
    z_est_index = np.argmax(prior)
    z_decode = z_candidates[z_est_index]
    return z_decode, phi_decode