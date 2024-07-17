import argparse
import numpy as np



parser = argparse.ArgumentParser()

# CPU / GPU setting
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--use_cuda', type=str, default='True')

# training hyperparameter
parser.add_argument('--epoch', type=int, default=20000)
parser.add_argument('--load_epoch', type=int, default=20000)

parser.add_argument('--repeat_num', type=int, default=5, help='The number of repeat experiment')
parser.add_argument('--coeff_range', type=int, default=10, help='The range of the coefficients')
parser.add_argument('--load_range', type=int, default=5, help='The range of the coefficients')

parser.add_argument('--pde_type', type=str, default='convection', help='The type of PDE')
parser.add_argument('--init_cond', type=str, default='sin_1', help='The initial condition')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--target_coeff', type=int, default=10, help='The range of the coefficients')
parser.add_argument('--hidden_dim', type=int, default=160, help='Hidden dimension of P^2INNs')

parser.add_argument('--beta', type=int, default = 1)
parser.add_argument('--nu', type=int, default = 1)
parser.add_argument('--rho', type=int, default = 1)

##############################################################################################################################################

def get_config():
    return parser.parse_args()


def get_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp







