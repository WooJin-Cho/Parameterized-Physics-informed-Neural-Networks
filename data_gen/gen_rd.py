import argparse
import numpy as np
import os
import random
import torch
from systems import *
import torch.backends.cudnn as cudnn
from utils import *
import matplotlib.pyplot as plt
import pdb
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--system', type=str, default='rd', help='System to study.')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--N_f', type=int, default=1000, help='Number of collocation points to sample.')
parser.add_argument('--L', type=float, default=1.0, help='Multiplier on loss f.')

parser.add_argument('--xgrid', type=int, default=256, help='Number of points in the xgrid.')
parser.add_argument('--nt', type=int, default=100, help='Number of points in the tgrid.')
parser.add_argument('--nu', type=float, default=1.0)
parser.add_argument('--rho', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--u0_str', default='1+sin(x)')
parser.add_argument('--source', default=0, type=float)

args = parser.parse_args()

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

beta = 0
nu = 0
rho = 0

for i in range(41):
    rho = 0
    for j in range(41):
        print('nu', nu, 'beta', beta, 'rho', rho)

        ############################
        # Process data
        ############################

        x = np.linspace(0, 2*np.pi, args.xgrid, endpoint=False).reshape(-1, 1) # not inclusive
        t = np.linspace(0, 1, args.nt).reshape(-1, 1)
        X, T = np.meshgrid(x, t) # all the X grid points T times, all the T grid points X times
        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])) # all the x,t "test" data
        t_noinitial = t[1:]
        x_noboundary = x[1:]
        X_noboundary, T_noinitial = np.meshgrid(x_noboundary, t_noinitial)
        X_star_noinitial_noboundary = np.hstack((X_noboundary.flatten()[:, None], T_noinitial.flatten()[:, None]))

        # sample collocation points only from the interior (where the PDE is enforced)
        X_f_train, idx_test, idx_val = sample_random(X_star_noinitial_noboundary, args.N_f)

        if 'convection' in args.system or 'diffusion' in args.system:
            u_vals, u_v = convection_diffusion(args.u0_str, nu, beta, args.source, args.xgrid, args.nt)
            G = np.full(X_f_train.shape[0], float(args.source))
            print(u_vals)
        elif 'rd' in args.system:
            u_vals, u_v = reaction_diffusion_discrete_solution(args.u0_str, nu, rho, args.xgrid, args.nt)
            G = np.full(X_f_train.shape[0], float(args.source))
        elif 'reaction' in args.system:
            u_vals = reaction_solution(args.u0_str, rho, args.xgrid, args.nt)
            G = np.full(X_f_train.shape[0], float(args.source))
        else:
            print("WARNING: System is not specified.")

        u_star = u_vals.reshape(-1, 1) # Exact solution reshaped into (n, 1)

        Exact = u_star.reshape(len(t), len(x)) # Exact on the (x,t) grid
        Exact_test = u_v[1:, 1:].flatten()[:, None]
        Exact_val = u_v[1:, 1:].flatten()[:, None]

        u_test = Exact_test[idx_test]
        X_test = X_star_noinitial_noboundary[idx_test, :]

        u_val = Exact_val[idx_val]
        X_val = X_star_noinitial_noboundary[idx_val, :]

        xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T)) # initial condition, from x = [-end, +end] and t=0
        uu1 = Exact[0:1,:].T # u(x, t) at t=0
        bc_lb = np.hstack((X[:,0:1], T[:,0:1])) # boundary condition at x = 0, and t = [0, 1]
        uu2 = Exact[:,0:1] # u(-end, t)

        # generate the other BC, now at x=2pi
        t = np.linspace(0, 1, args.nt).reshape(-1, 1)
        x_bc_ub = np.array([2*np.pi]*t.shape[0]).reshape(-1, 1)
        bc_ub = np.hstack((x_bc_ub, t))

        u_train = uu1 # just the initial condition
        X_u_train = xx1 # (x,t) for initial condition

        X_f_beta    = np.array([beta] * len(X_f_train)).reshape(-1, 1)
        X_f_nu      = np.array([nu] * len(X_f_train)).reshape(-1, 1)
        X_f_rho     = np.array([rho] * len(X_f_train)).reshape(-1, 1)

        X_u_beta    = np.array([beta] * len(X_u_train)).reshape(-1, 1)
        X_u_nu      = np.array([nu] * len(X_u_train)).reshape(-1, 1)
        X_u_rho     = np.array([rho] * len(X_u_train)).reshape(-1, 1)

        X_test_beta = np.array([beta] * len(X_test)).reshape(-1, 1)
        X_test_nu   = np.array([nu] * len(X_test)).reshape(-1, 1)
        X_test_rho  = np.array([rho] * len(X_test)).reshape(-1, 1)

        X_val_beta = np.array([beta] * len(X_val)).reshape(-1, 1)
        X_val_nu   = np.array([nu] * len(X_val)).reshape(-1, 1)
        X_val_rho  = np.array([rho] * len(X_val)).reshape(-1, 1)

        X_bd_beta   = np.array([beta] * len(bc_lb)).reshape(-1, 1)
        X_bd_nu     = np.array([nu] * len(bc_lb)).reshape(-1, 1)
        X_bd_rho    = np.array([rho] * len(bc_lb)).reshape(-1, 1)

        dummy       = np.array([0] * len(X_f_train)).reshape(-1, 1)

        X_train_u   = np.concatenate((X_u_train, u_train, X_u_beta, X_u_nu, X_u_rho), axis=1)
        X_train_f   = np.concatenate((X_f_train, dummy, X_f_beta, X_f_nu, X_f_rho), axis=1)
        X_test_arr  = np.concatenate((X_test, u_test, X_test_beta, X_test_nu, X_test_rho), axis=1)
        X_val       = np.concatenate((X_val, u_val, X_val_beta, X_val_nu, X_val_rho), axis=1)

        X_boundary  = np.concatenate((bc_lb, bc_ub, X_bd_beta, X_bd_nu, X_bd_rho), axis=1)


        X_train_u_df    = pd.DataFrame(X_train_u)
        X_train_f_df    = pd.DataFrame(X_train_f)
        X_test_df       = pd.DataFrame(X_test_arr)
        X_val_df        = pd.DataFrame(X_val)
        X_boundary_df   = pd.DataFrame(X_boundary)


        X_train_u_df.columns    = ['x_data', 't_data', 'u_data', 'beta', 'nu', 'rho']
        X_train_f_df.columns    = ['x_data', 't_data', 'u_data', 'beta', 'nu', 'rho']
        X_boundary_df.columns   = ['x_data_lb', 't_data_lb', 'x_data_ub', 't_data_ub', 'beta', 'nu', 'rho']
        X_test_df.columns       = ['x_data', 't_data', 'u_data', 'beta', 'nu', 'rho']
        X_val_df.columns        = ['x_data', 't_data', 'u_data', 'beta', 'nu', 'rho']


        train_boundary_file_name = './dataset/rd/train/train_boundary_' + str(int(nu)) + '_' + str(int(rho)) + '_rd.csv'
        train_u_file_name        = './dataset/rd/train/train_u_' + str(int(nu)) + '_' + str(int(rho)) + '_rd.csv'
        train_f_file_name        = './dataset/rd/train/train_f_' + str(int(nu)) + '_' + str(int(rho)) + '_rd.csv'
        test_file_name           = './dataset/rd/test/test_' + str(int(nu)) + '_' + str(int(rho)) + '_rd.csv'
        val_file_name            = './dataset/rd/val/val_' + str(int(nu)) + '_' + str(int(rho)) + '_rd.csv'


        X_boundary_df.to_csv(train_boundary_file_name, mode='w', index=False)
        X_train_u_df.to_csv(train_u_file_name, mode='w', index=False)
        X_train_f_df.to_csv(train_f_file_name, mode='w', index=False)
        X_test_df.to_csv(test_file_name, mode='w', index=False)
        X_val_df.to_csv(val_file_name, mode='w', index=False)

        rho += 1
    nu += 1
