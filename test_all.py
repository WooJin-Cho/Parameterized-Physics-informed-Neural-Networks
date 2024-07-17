import torch
from config import get_config, get_params
from model import P2INN_phase1
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
from sklearn.metrics import explained_variance_score, max_error
import utils

def test(net, test_dataloader, device):
    with torch.autograd.no_grad():
        net.eval()

        u_pred_test_list = []
        u_test_list = []
        
        for samples_test in test_dataloader:

            x_test, t_test, u_test, beta_test, nu_test, rho_test, eq_test = samples_test
            
            x_test      = x_test.clone().detach().requires_grad_(True).to(device)
            t_test      = t_test.clone().detach().requires_grad_(True).to(device)
            u_test      = u_test.clone().detach().requires_grad_(True).to(device)
            beta_test   = beta_test.clone().detach().requires_grad_(True).to(device)
            nu_test     = nu_test.clone().detach().requires_grad_(True).to(device)
            rho_test    = rho_test.clone().detach().requires_grad_(True).to(device)
            eq_test     = eq_test.to(device).float()
            
            u_pred_test = net(eq_test, x_test, t_test)
            
            if len(u_pred_test_list) == 0:
                u_pred_test_list = u_pred_test[:, 0]
                u_test_list = u_test[:, 0]
                
            else:
                u_pred_test_list = torch.cat((u_pred_test_list, u_pred_test[:, 0]), dim=0)
                u_test_list = torch.cat((u_test_list, u_test[:, 0]), dim=0)
                
        u_pred_test_tensor = u_pred_test_list
        u_test_tensor = u_test_list

        L2_error_norm = torch.linalg.norm(u_pred_test_tensor-u_test_tensor, 2, dim = 0)
        L2_true_norm = torch.linalg.norm(u_test_tensor, 2, dim = 0)

        L2_absolute_error = torch.mean(torch.abs(u_pred_test_tensor-u_test_tensor))
        L2_relative_error = L2_error_norm / L2_true_norm
        
        u_test_tensor = u_test_tensor.cpu()
        u_pred_test_tensor = u_pred_test_tensor.cpu()
        
        Max_err = max_error(u_test_tensor, u_pred_test_tensor)
        Ex_var_score = explained_variance_score(u_test_tensor, u_pred_test_tensor)
        
        return L2_absolute_error.item(), L2_relative_error.item(), Max_err, Ex_var_score

def main():
    args=get_config()
    random_seed = args.seed
    
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(random_seed)
    
    device = torch.device(args.device)

    epoch = args.epoch
    pde_type = args.pde_type
    initial_condition = args.init_cond
    coefficient_range = args.coeff_range
    coeff1, coeff2, coeff3 = coefficient_range, coefficient_range, coefficient_range
    
    PATH = f'./param/{initial_condition}/{pde_type}_{coefficient_range}_{random_seed}/P2INN_{epoch}.pt'
    
    if "checkpoint" in pde_type:
        pde_type = pde_type.split('checkpoint_')[1]
    
    net = P2INN_phase1().to(device)
    net.load_state_dict(torch.load(PATH))
    model_size = get_params(net)

    print("=============[Test Info]===============")
    print(f"- PDE type : all types")
    print(f"- Initial condition : {initial_condition}")
    print(f"- Coefficient range : {coeff1}, {coeff2}, {coeff3}")
    print(f"- Model size : {model_size}")
    print("========================================\n")
    
    pde_types = ['convection', 'diffusion', 'reaction', 'cd', 'rd', 'cdr']
        
    print(f"    Type     |   Abs      Rel    Max_err  Exp_var")
    
    for pde_type in pde_types:
        
        l2_abs_errors, l2_rel_errors, max_errs, ex_vars = [], [], [], []
        
        if pde_type in ['convection', 'diffusion', 'reaction']:            
            for i in range(1, coeff1+1):
                test_dataloader = utils.get_dataloader_by_type_by_coeff_for_test(initial_condition, pde_type, i, 0, 0)
                l2_abs, l2_rel, max_err, ex_var = test(net, test_dataloader, device)
                l2_abs_errors.append(l2_abs)
                l2_rel_errors.append(l2_rel)
                max_errs.append(max_err)
                ex_vars.append(ex_var)

        elif pde_type in ['cd', 'rd']:
            for i in range(1, coeff1+1):
                for j in range(1, coeff2+1):
                    test_dataloader = utils.get_dataloader_by_type_by_coeff_for_test(initial_condition, pde_type, i, j, 0)
                    l2_abs, l2_rel, max_err, ex_var = test(net, test_dataloader, device)
                    l2_abs_errors.append(l2_abs)
                    l2_rel_errors.append(l2_rel)
                    max_errs.append(max_err)
                    ex_vars.append(ex_var)

        elif pde_type == 'cdr':
            for i in range(1, coeff1+1):
                for j in range(1, coeff2+1):
                    for k in range(1, coeff3+1):
                        test_dataloader = utils.get_dataloader_by_type_by_coeff_for_test(initial_condition, pde_type, i, j, k)
                        l2_abs, l2_rel, max_err, ex_var = test(net, test_dataloader, device)
                        l2_abs_errors.append(l2_abs)
                        l2_rel_errors.append(l2_rel)
                        max_errs.append(max_err)
                        ex_vars.append(ex_var)
        
        
        print(f"- {pde_type: <10} : {np.mean(np.array(l2_abs_errors)):.4f}, {np.mean(np.array(l2_rel_errors)):.4f}, {np.mean(np.array(max_errs)):.4f}, {np.mean(np.array(ex_vars)):.4f}")


if __name__ == "__main__":
    main()


