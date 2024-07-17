import pandas as pd
import torch
import numpy as np
import torch
from torch.utils.data import Dataset
 
class P2INN_Dataset(Dataset):
    def __init__(self, df):

        self.df = df
        self.x_data = self.df['x_data'].values.reshape(len(self.df), 1)
        self.t_data = self.df['t_data'].values.reshape(len(self.df), 1)
        self.u_data = self.df['u_data'].values.reshape(len(self.df), 1)
        self.beta_data = self.df['beta'].values.reshape(len(self.df), 1)
        self.nu_data = self.df['nu'].values.reshape(len(self.df), 1)
        self.rho_data = self.df['rho'].values.reshape(len(self.df), 1)

        self.eq_data = []
        for i in range(len(self.df)):
            beta    = self.beta_data[i]
            nu      = self.nu_data[i]
            rho     = self.rho_data[i]

            General_form = [int(beta), int(nu), int(rho)]              
            self.eq_data.append(General_form)

        self.eq_data = np.array(self.eq_data)
        self.eq_data = pd.DataFrame(self.eq_data)
        self.eq_data = self.eq_data.values


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        x_data = torch.FloatTensor(self.x_data[idx])
        t_data = torch.FloatTensor(self.t_data[idx])
        u_data = torch.FloatTensor(self.u_data[idx])
        beta_data = torch.FloatTensor(self.beta_data[idx])
        nu_data = torch.FloatTensor(self.nu_data[idx])
        rho_data = torch.FloatTensor(self.rho_data[idx])
        eq_data = torch.LongTensor(self.eq_data[idx])

        return x_data, t_data, u_data, beta_data, nu_data, rho_data, eq_data



class P2INN_Dataset_bd(Dataset):

    def __init__(self, df):
        self.df = df
        self.x_data_lb = self.df['x_data_lb'].values.reshape(len(self.df), 1)
        self.t_data_lb = self.df['t_data_lb'].values.reshape(len(self.df), 1)
        self.x_data_ub = self.df['x_data_ub'].values.reshape(len(self.df), 1)
        self.t_data_ub = self.df['t_data_ub'].values.reshape(len(self.df), 1)

        self.beta_data = self.df['beta'].values.reshape(len(self.df), 1)
        self.nu_data = self.df['nu'].values.reshape(len(self.df), 1)
        self.rho_data = self.df['rho'].values.reshape(len(self.df), 1)

        self.eq_data = []
        for i in range(len(self.df)):
            beta    = self.beta_data[i]
            nu      = self.nu_data[i]
            rho     = self.rho_data[i]

            General_form = [int(beta), int(nu), int(rho)]              
            self.eq_data.append(General_form)


        self.eq_data = np.array(self.eq_data)
        self.eq_data = pd.DataFrame(self.eq_data)
        self.eq_data = self.eq_data.values


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        x_data_lb = torch.FloatTensor(self.x_data_lb[idx])
        t_data_lb = torch.FloatTensor(self.t_data_lb[idx])

        x_data_ub = torch.FloatTensor(self.x_data_ub[idx])
        t_data_ub = torch.FloatTensor(self.t_data_ub[idx])

        beta_data = torch.FloatTensor(self.beta_data[idx])
        nu_data = torch.FloatTensor(self.nu_data[idx])
        rho_data = torch.FloatTensor(self.rho_data[idx])
        eq_data = torch.LongTensor(self.eq_data[idx])

        return x_data_lb, t_data_lb, x_data_ub, t_data_ub, beta_data, nu_data, rho_data, eq_data
