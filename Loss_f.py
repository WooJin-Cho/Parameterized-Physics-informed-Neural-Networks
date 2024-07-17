import torch
from config import get_config


def PDE_cal(eq, x, t, beta, nu, rho, net):
    args = get_config()
    u = net(eq, x, t) 

    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u), 
        retain_graph=True, 
        create_graph=True)[0]
        
    u_xx = torch.autograd.grad(
        u_x, x, 
        grad_outputs=torch.ones_like(u_x), 
        retain_graph=True, 
        create_graph=True)[0]

    u_t = torch.autograd.grad(
        u, t, 
        grad_outputs=torch.ones_like(u), 
        retain_graph=True, 
        create_graph=True)[0]

    pde = (beta * u_x) - (nu * u_xx) - (rho * u * (1 - u)) + u_t 

    return pde
