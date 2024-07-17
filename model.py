import torch
import torch.nn as nn
from config import get_config

args=get_config()

class P2INN_phase1(nn.Module):
    def __init__(self):
        super(P2INN_phase1, self).__init__()
        
        # model encoder 
        self.param_enc_layer_1 = nn.Linear(3, 150)
        self.param_enc_layer_2 = nn.Linear(150, 150)
        self.param_enc_layer_3 = nn.Linear(150, 150)
        self.param_enc_layer_4 = nn.Linear(150, 50)
        
        # model decoder
        self.coord_enc_layer_1 = nn.Linear(2, 50)
        self.coord_enc_layer_2 = nn.Linear(50, 50)
        self.coord_enc_layer_3 = nn.Linear(50, 50)
        
        self.tanh       = nn.Tanh()
        self.relu       = nn.ReLU(inplace=True)

        self.dec_layer_1 = nn.Linear(100, 50)
        self.dec_layer_2 = nn.Linear(50, 50)
        self.dec_layer_3 = nn.Linear(50, 50)
        self.dec_layer_4 = nn.Linear(50, 50)
        self.dec_layer_5 = nn.Linear(50, 50)
        self.dec_layer_6 = nn.Linear(50, 50)
        self.last_layer = nn.Linear(50, 1)
        
        self.modvec = nn.Linear(5, 50)



    def forward(self, eq, x, t):
        
        ## ENCODER ##
        param_embed = self.param_enc_layer_1(eq)
        param_embed = self.tanh(param_embed)
        param_embed = self.param_enc_layer_2(param_embed)
        param_embed = self.tanh(param_embed)
        param_embed = self.param_enc_layer_3(param_embed)
        param_embed = self.tanh(param_embed)
        param_embed = self.param_enc_layer_4(param_embed)

        coord_inputs = torch.cat([x, t], axis=1) 

        coord_embed = self.coord_enc_layer_1(coord_inputs)  
        coord_embed = self.tanh(coord_embed)
        coord_embed = self.coord_enc_layer_2(coord_embed) 
        coord_embed = self.tanh(coord_embed)
        coord_embed = self.coord_enc_layer_3(coord_embed)
        coord_embed = self.tanh(coord_embed)

        z_vector = torch.concat([param_embed, coord_embed], axis = 1) 

        ## DECODER ##
        dec_emb_1 = self.dec_layer_1(z_vector)
        dec_emb_1 = self.relu(dec_emb_1)
        
        dec_emb_2 = self.dec_layer_2(dec_emb_1)
        dec_emb_2 = self.relu(dec_emb_2)

        dec_emb_3 = self.dec_layer_3(dec_emb_2) + dec_emb_1
        dec_emb_3 = self.relu(dec_emb_3)     

        dec_emb_4 = self.dec_layer_4(dec_emb_3) + dec_emb_2
        dec_emb_4 = self.relu(dec_emb_4)   

        dec_emb_5 = self.dec_layer_5(dec_emb_4) + dec_emb_3
        dec_emb_5 = self.relu(dec_emb_5)  
        
        input_vector = torch.concat([eq, coord_inputs], axis = 1) 
        hyper_shift = self.modvec(input_vector)    
        
        dec_emb_6 = self.dec_layer_6(dec_emb_5) + dec_emb_4 + hyper_shift
        dec_emb_6 = self.tanh(dec_emb_6)   

        pred = self.last_layer(dec_emb_6)

        return pred
    


class P2INN_phase2_svd(nn.Module):
    def __init__(self, cols, rows, sigmas, biases):
        super(P2INN_phase2_svd, self).__init__()
        
        col_2, col_3, col_4, col_5, col_6 = cols
        row_2, row_3, row_4, row_5, row_6 = rows
        sigma_2, sigma_3, sigma_4, sigma_5, sigma_6 = sigmas
        bias_2, bias_3, bias_4, bias_5, bias_6 = biases
        
        # model encoder 
        self.param_enc_layer_1 = nn.Linear(3, 150)
        self.param_enc_layer_2 = nn.Linear(150, 150)
        self.param_enc_layer_3 = nn.Linear(150, 150)
        self.param_enc_layer_4 = nn.Linear(150, 50)
        
        # model decoder
        self.coord_enc_layer_1 = nn.Linear(2, 50)
        self.coord_enc_layer_2 = nn.Linear(50, 50)
        self.coord_enc_layer_3 = nn.Linear(50, 50)
        
        self.tanh       = nn.Tanh()
        self.relu       = nn.ReLU(inplace=True)

        self.dec_layer_1 = nn.Linear(100, 50)
        self.last_layer = nn.Linear(50, 1)
        
        self.col_basis_2 = nn.Parameter(col_2, requires_grad=False)
        self.col_basis_3 = nn.Parameter(col_3, requires_grad=False)
        self.col_basis_4 = nn.Parameter(col_4, requires_grad=False)
        self.col_basis_5 = nn.Parameter(col_5, requires_grad=False)
        self.col_basis_6 = nn.Parameter(col_6, requires_grad=False)
        
        self.row_basis_2 = nn.Parameter(row_2, requires_grad=False)
        self.row_basis_3 = nn.Parameter(row_3, requires_grad=False)
        self.row_basis_4 = nn.Parameter(row_4, requires_grad=False)
        self.row_basis_5 = nn.Parameter(row_5, requires_grad=False)
        self.row_basis_6 = nn.Parameter(row_6, requires_grad=False)
        
        self.sigma_2 = nn.Parameter(sigma_2)
        self.sigma_3 = nn.Parameter(sigma_3)
        self.sigma_4 = nn.Parameter(sigma_4)
        self.sigma_5 = nn.Parameter(sigma_5)
        self.sigma_6 = nn.Parameter(sigma_6)
        
        self.bias_2 = nn.Parameter(bias_2, requires_grad=False)
        self.bias_3 = nn.Parameter(bias_3, requires_grad=False)
        self.bias_4 = nn.Parameter(bias_4, requires_grad=False)
        self.bias_5 = nn.Parameter(bias_5, requires_grad=False)
        self.bias_6 = nn.Parameter(bias_6, requires_grad=False)
        
        self.modvec = nn.Linear(5, 50)

    def forward(self, eq, x, t):
        
        ## ENCODER ##
        param_embed = self.param_enc_layer_1(eq)
        param_embed = self.tanh(param_embed)
        param_embed = self.param_enc_layer_2(param_embed)
        param_embed = self.tanh(param_embed)
        param_embed = self.param_enc_layer_3(param_embed)
        param_embed = self.tanh(param_embed)
        param_embed = self.param_enc_layer_4(param_embed)

        coord_inputs = torch.cat([x, t], axis=1) 

        coord_embed = self.coord_enc_layer_1(coord_inputs)  
        coord_embed = self.tanh(coord_embed)
        coord_embed = self.coord_enc_layer_2(coord_embed) 
        coord_embed = self.tanh(coord_embed)
        coord_embed = self.coord_enc_layer_3(coord_embed)
        coord_embed = self.tanh(coord_embed)

        z_vector = torch.concat([param_embed, coord_embed], axis = 1) 

        ## DECODER ##
        dec_emb_1 = self.dec_layer_1(z_vector)
        dec_emb_1 = self.relu(dec_emb_1)
        
        weight_2 = torch.mm(torch.mm(self.col_basis_2, torch.diag(self.sigma_2)), self.row_basis_2)
        dec_emb_2 = torch.matmul(dec_emb_1, weight_2.t()) + self.bias_2
        dec_emb_2 = self.relu(dec_emb_2)

        weight_3 = torch.mm(torch.mm(self.col_basis_3, torch.diag(self.sigma_3)), self.row_basis_3)
        dec_emb_3 = torch.matmul(dec_emb_2, weight_3.t()) + self.bias_3 + dec_emb_1
        dec_emb_3 = self.relu(dec_emb_3)        

        weight_4 = torch.mm(torch.mm(self.col_basis_4, torch.diag(self.sigma_4)), self.row_basis_4)
        dec_emb_4 = torch.matmul(dec_emb_3, weight_4.t()) + self.bias_4 + dec_emb_2
        dec_emb_4 = self.relu(dec_emb_4)      

        weight_5 = torch.mm(torch.mm(self.col_basis_5, torch.diag(self.sigma_5)), self.row_basis_5)
        dec_emb_5 = torch.matmul(dec_emb_4, weight_5.t()) + self.bias_5 + dec_emb_3
        dec_emb_5 = self.relu(dec_emb_5)
        
        input_vector = torch.concat([eq, coord_inputs], axis = 1) 
        hyper_shift = self.modvec(input_vector)    
        
        weight_6 = torch.mm(torch.mm(self.col_basis_6, torch.diag(self.sigma_6)), self.row_basis_6)
        dec_emb_6 = torch.matmul(dec_emb_5, weight_6.t()) + self.bias_6 + dec_emb_4 + hyper_shift
        dec_emb_6 = self.tanh(dec_emb_6)   

        pred = self.last_layer(dec_emb_6)

        return pred
    