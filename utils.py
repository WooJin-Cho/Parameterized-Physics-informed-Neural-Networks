from dataloader import P2INN_Dataset, P2INN_Dataset_bd
from torch.utils.data import DataLoader
import pandas as pd

def get_dataloader_all_types_w_bd(initial_condition, coefficient_range):

    # Equation dataset
    pde_type = 'convection'
    
    train_data_f        = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_f_1_'+str(pde_type)+'.csv')
    train_data_u        = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_u_1_'+str(pde_type)+'.csv')
    train_data_bd       = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_boundary_1_'+str(pde_type)+'.csv')
    test_data           = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/test/test_1_'+str(pde_type)+'.csv')

    for pde_type in ['convection']:
        for i in range(1, coefficient_range):
            f_sample    = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_f_'+str(i+1)+'_'+str(pde_type)+'.csv')
            u_sample    = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_u_'+str(i+1)+'_'+str(pde_type)+'.csv')
            bd_sample   = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_boundary_'+str(i+1)+'_'+str(pde_type)+'.csv')
            test_sample = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/test/test_' + str(i+1) + '_'+str(pde_type)+'.csv')

            train_data_f    = pd.concat([train_data_f, f_sample], ignore_index = True)
            train_data_u    = pd.concat([train_data_u, u_sample], ignore_index = True)
            train_data_bd   = pd.concat([train_data_bd, bd_sample], ignore_index = True)
            test_data       = pd.concat([test_data, test_sample], ignore_index = True)

    for pde_type in ['diffusion', 'reaction']:
        for i in range(0, coefficient_range):
            f_sample    = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_f_'+str(i+1)+'_'+str(pde_type)+'.csv')
            u_sample    = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_u_'+str(i+1)+'_'+str(pde_type)+'.csv')
            bd_sample   = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_boundary_'+str(i+1)+'_'+str(pde_type)+'.csv')
            test_sample = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/test/test_' + str(i+1) + '_'+str(pde_type)+'.csv')

            train_data_f    = pd.concat([train_data_f, f_sample], ignore_index = True)
            train_data_u    = pd.concat([train_data_u, u_sample], ignore_index = True)
            train_data_bd   = pd.concat([train_data_bd, bd_sample], ignore_index = True)
            test_data       = pd.concat([test_data, test_sample], ignore_index = True)
            
    for pde_type in ['convection_diffusion', 'reaction_diffusion']:
        for i in range(0, coefficient_range):
            for j in range(0, coefficient_range):
                
                if i==0 and j==0:
                    continue
                            
                f_sample    = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_f_'+str(i+1)+'_'+str(j+1)+'_'+str(pde_type)+'.csv')
                u_sample    = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_u_'+str(i+1)+'_'+str(j+1)+'_'+str(pde_type)+'.csv')
                bd_sample   = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_boundary_'+str(i+1)+'_'+str(j+1)+'_'+str(pde_type)+'.csv')
                test_sample = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/test/test_'+str(i+1)+'_'+str(j+1)+'_'+str(pde_type)+'.csv')

                train_data_f    = pd.concat([train_data_f, f_sample], ignore_index = True)
                train_data_u    = pd.concat([train_data_u, u_sample], ignore_index = True)
                train_data_bd   = pd.concat([train_data_bd, bd_sample], ignore_index = True)
                test_data       = pd.concat([test_data, test_sample], ignore_index = True)

    for pde_type in ['convection_diffusion_reaction']:
        for i in range(0, coefficient_range):
            for j in range(0, coefficient_range):
                for k in range(0, coefficient_range):
                    if i==0 and j==0 and k==0:
                        continue

                    f_sample    = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_f_'+str(i+1)+'_'+str(j+1)+'_'+str(k+1)+'_'+str(pde_type)+'.csv')
                    u_sample    = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_u_'+str(i+1)+'_'+str(j+1)+'_'+str(k+1)+'_'+str(pde_type)+'.csv')
                    bd_sample   = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_boundary_'+str(i+1)+'_'+str(j+1)+'_'+str(k+1)+'_'+str(pde_type)+'.csv')
                    test_sample = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/test/test_'+str(i+1)+'_'+str(j+1)+'_'+str(k+1)+'_'+str(pde_type)+'.csv')

                    train_data_f    = pd.concat([train_data_f, f_sample], ignore_index = True)
                    train_data_u    = pd.concat([train_data_u, u_sample], ignore_index = True)
                    train_data_bd   = pd.concat([train_data_bd, bd_sample], ignore_index = True)
                    test_data       = pd.concat([test_data, test_sample], ignore_index = True)
                    
    f_dataset = P2INN_Dataset(df=train_data_f)
    u_dataset = P2INN_Dataset(df=train_data_u)
    bd_dataset = P2INN_Dataset_bd(df=train_data_bd)
    test_dataset = P2INN_Dataset(df=test_data)


    f_dataloader = DataLoader(f_dataset, batch_size=20000, num_workers=4, shuffle=True)
    u_dataloader = DataLoader(u_dataset, batch_size=20000, num_workers=4, shuffle=True)
    bd_dataloader = DataLoader(bd_dataset, batch_size=20000, num_workers=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=20000, num_workers=4, shuffle=True)
        
    return f_dataloader, u_dataloader, bd_dataloader, test_dataloader
    
   
def get_dataloader_by_type_w_bd(initial_condition, pde_type, coefficient_range):
    
    if pde_type in ['convection', 'diffusion', 'reaction']:
        # Equation dataset
        train_data_f        = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_f_1_'+str(pde_type)+'.csv')
        train_data_u        = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_u_1_'+str(pde_type)+'.csv')
        train_data_bd       = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_boundary_1_'+str(pde_type)+'.csv')
        test_data           = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/test/test_1_'+str(pde_type)+'.csv')

        for i in range(1, coefficient_range):
            f_sample    = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_f_'+str(i+1)+'_'+str(pde_type)+'.csv')
            u_sample    = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_u_'+str(i+1)+'_'+str(pde_type)+'.csv')
            bd_sample   = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_boundary_'+str(i+1)+'_'+str(pde_type)+'.csv')
            test_sample = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/test/test_' + str(i+1) + '_'+str(pde_type)+'.csv')

            train_data_f    = pd.concat([train_data_f, f_sample], ignore_index = True)
            train_data_u    = pd.concat([train_data_u, u_sample], ignore_index = True)
            train_data_bd   = pd.concat([train_data_bd, bd_sample], ignore_index = True)
            test_data       = pd.concat([test_data, test_sample], ignore_index = True)
            

    elif pde_type in ['convection_diffusion', 'reaction_diffusion']:
        # Equation dataset
        train_data_f        = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_f_1_1_'+str(pde_type)+'.csv')
        train_data_u        = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_u_1_1_'+str(pde_type)+'.csv')
        train_data_bd       = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_boundary_1_1_'+str(pde_type)+'.csv')
        test_data           = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/test/test_1_1_'+str(pde_type)+'.csv')

        for i in range(0, coefficient_range):
            for j in range(0, coefficient_range):
                
                if i==0 and j==0:
                    continue
                            
                f_sample    = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_f_'+str(i+1)+'_'+str(j+1)+'_'+str(pde_type)+'.csv')
                u_sample    = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_u_'+str(i+1)+'_'+str(j+1)+'_'+str(pde_type)+'.csv')
                bd_sample   = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_boundary_'+str(i+1)+'_'+str(j+1)+'_'+str(pde_type)+'.csv')
                test_sample = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/test/test_'+str(i+1)+'_'+str(j+1)+'_'+str(pde_type)+'.csv')

                train_data_f    = pd.concat([train_data_f, f_sample], ignore_index = True)
                train_data_u    = pd.concat([train_data_u, u_sample], ignore_index = True)
                train_data_bd   = pd.concat([train_data_bd, bd_sample], ignore_index = True)
                test_data       = pd.concat([test_data, test_sample], ignore_index = True)



    elif pde_type == 'convection_diffusion_reaction':
        # Equation dataset
        train_data_f        = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_f_1_1_1_'+str(pde_type)+'.csv')
        train_data_u        = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_u_1_1_1_'+str(pde_type)+'.csv')
        train_data_bd       = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_boundary_1_1_1_'+str(pde_type)+'.csv')
        test_data           = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/test/test_1_1_1_'+str(pde_type)+'.csv')


        for i in range(0, coefficient_range):
            for j in range(0, coefficient_range):
                for k in range(0, coefficient_range):
                    if i==0 and j==0 and k==0:
                        continue

                    f_sample    = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_f_'+str(i+1)+'_'+str(j+1)+'_'+str(k+1)+'_'+str(pde_type)+'.csv')
                    u_sample    = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_u_'+str(i+1)+'_'+str(j+1)+'_'+str(k+1)+'_'+str(pde_type)+'.csv')
                    bd_sample   = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/train/train_boundary_'+str(i+1)+'_'+str(j+1)+'_'+str(k+1)+'_'+str(pde_type)+'.csv')
                    test_sample = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/test/test_'+str(i+1)+'_'+str(j+1)+'_'+str(k+1)+'_'+str(pde_type)+'.csv')

                    train_data_f    = pd.concat([train_data_f, f_sample], ignore_index = True)
                    train_data_u    = pd.concat([train_data_u, u_sample], ignore_index = True)
                    train_data_bd   = pd.concat([train_data_bd, bd_sample], ignore_index = True)
                    test_data       = pd.concat([test_data, test_sample], ignore_index = True)


    f_dataset = P2INN_Dataset(df=train_data_f)
    u_dataset = P2INN_Dataset(df=train_data_u)
    bd_dataset = P2INN_Dataset_bd(df=train_data_bd)
    test_dataset = P2INN_Dataset(df=test_data)
    

    f_dataloader = DataLoader(f_dataset, batch_size=20000, shuffle=True)
    u_dataloader = DataLoader(u_dataset, batch_size=20000, shuffle=True)
    bd_dataloader = DataLoader(bd_dataset, batch_size=20000, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=20000, shuffle=True)
    
    return f_dataloader, u_dataloader, bd_dataloader, test_dataloader

  
def get_dataloader_by_type_by_coeff_for_test(initial_condition, pde_type, coeff1, coeff2, coeff3):
    
    if pde_type in ['convection', 'diffusion', 'reaction']:
        test_data           = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/test/test_{coeff1}_'+str(pde_type)+'.csv')

    elif pde_type in ['cd', 'rd']:
        test_data           = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/test/test_{coeff1}_{coeff2}_'+str(pde_type)+'.csv')

    elif pde_type == 'cdr':
        test_data           = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/test/test_{coeff1}_{coeff2}_{coeff3}_'+str(pde_type)+'.csv')
   
    test_dataset = P2INN_Dataset(df=test_data)    
    test_dataloader = DataLoader(test_dataset, batch_size=20000, num_workers=3, shuffle=True)
    
    return  test_dataloader

def get_dataloader_by_type_for_test(initial_condition, pde_type, coeff1, coeff2, coeff3):
    
    if pde_type in ['convection', 'diffusion', 'reaction']:
        test_data           = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/test/test_1_'+str(pde_type)+'.csv')
        
        for i in range(1, coeff1):
            test_sample = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/test/test_' + str(i+1) + '_'+str(pde_type)+'.csv')
            test_data       = pd.concat([test_data, test_sample], ignore_index = True)

    elif pde_type in ['convection_diffusion', 'reaction_diffusion']:
        test_data           = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/test/test_1_1_'+str(pde_type)+'.csv')
        
        for i in range(0, coeff1):
            for j in range(0, coeff2):
                
                if i==0 and j==0:
                    continue
                        
                test_sample = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/test/test_'+str(i+1)+'_'+str(j+1)+'_'+str(pde_type)+'.csv')
                test_data       = pd.concat([test_data, test_sample], ignore_index = True)

    elif pde_type == 'convection_diffusion_reaction':
        test_data           = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/test/test_1_1_1_'+str(pde_type)+'.csv')
        
        for i in range(0, coeff1):
            for j in range(0, coeff2):
                for k in range(0, coeff3):
                    if i==0 and j==0 and k==0:
                        continue

                    test_sample = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+'/test/test_'+str(i+1)+'_'+str(j+1)+'_'+str(k+1)+'_'+str(pde_type)+'.csv')
                    test_data   = pd.concat([test_data, test_sample], ignore_index = True)
        
    test_dataset = P2INN_Dataset(df=test_data)    
    test_dataloader = DataLoader(test_dataset, batch_size=20000, num_workers=3, shuffle=True)
    
    return  test_dataloader

def get_dataloader_by_type_w_bd_target(initial_condition, pde_type, coefficient_range):

    if pde_type in ['convection', 'diffusion', 'reaction']:
        # Equation dataset
        train_data_f        = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/train/train_f_{coefficient_range}_'+str(pde_type)+'.csv')
        train_data_u        = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/train/train_u_{coefficient_range}_'+str(pde_type)+'.csv')
        train_data_bd       = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/train/train_boundary_{coefficient_range}_'+str(pde_type)+'.csv') 
        test_data           = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/test/test_{coefficient_range}_'+str(pde_type)+'.csv')
        
    elif pde_type in ['reaction_diffusion', 'convection_diffusion']:
        # Equation dataset
        coeff1, coeff2 = coefficient_range
        train_data_f        = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/train/train_f_{coeff1}_{coeff2}_'+str(pde_type)+'.csv')
        train_data_u        = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/train/train_u_{coeff1}_{coeff2}_'+str(pde_type)+'.csv')
        train_data_bd       = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/train/train_boundary_{coeff1}_{coeff2}_'+str(pde_type)+'.csv')
        test_data           = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/test/test_{coeff1}_{coeff2}_'+str(pde_type)+'.csv')

    f_dataset = P2INN_Dataset(df=train_data_f)
    u_dataset = P2INN_Dataset(df=train_data_u)
    bd_dataset = P2INN_Dataset_bd(df=train_data_bd)
    test_dataset = P2INN_Dataset(df=test_data)

    f_dataloader = DataLoader(f_dataset, batch_size=20000, shuffle=True)
    u_dataloader = DataLoader(u_dataset, batch_size=20000, shuffle=True)
    bd_dataloader = DataLoader(bd_dataset, batch_size=20000, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=20000, shuffle=True)
    
    return f_dataloader, u_dataloader, bd_dataloader, test_dataloader



def get_dataloader_only_one_w_bd(initial_condition, pde_type, beta, nu, rho):
    
    assert initial_condition != 'tanh', "Current initial condition doesn't satisfy the boundary condition."
    
    if pde_type in ['convection', 'diffusion', 'reaction']:
        
        if pde_type == 'convection': coeff1 = beta
        elif pde_type == 'diffusion': coeff1 = nu
        elif pde_type == 'reaction': coeff1 = rho
        
        train_data_f        = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/train/train_f_{coeff1}_'+str(pde_type)+'.csv')
        train_data_u        = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/train/train_u_{coeff1}_'+str(pde_type)+'.csv')
        train_data_bd       = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/train/train_boundary_{coeff1}_'+str(pde_type)+'.csv') 
        test_data           = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/test/test_{coeff1}_'+str(pde_type)+'.csv')
        

    elif pde_type in ['cd', 'rd']:
        if pde_type == 'cd':
            coeff1 = beta
            coeff2 = nu
        elif pde_type == 'rd':
            coeff1 = nu
            coeff2 = rho

        train_data_f        = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/train/train_f_{coeff1}_{coeff2}_'+str(pde_type)+'.csv')
        train_data_u        = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/train/train_u_{coeff1}_{coeff2}_'+str(pde_type)+'.csv')
        train_data_bd       = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/train/train_boundary_{coeff1}_{coeff2}_'+str(pde_type)+'.csv')
        test_data           = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/test/test_{coeff1}_{coeff2}_'+str(pde_type)+'.csv')


    elif pde_type == 'cdr':
        train_data_f        = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/train/train_f_{beta}_{nu}_{rho}_'+str(pde_type)+'.csv')
        train_data_u        = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/train/train_u_{beta}_{nu}_{rho}_'+str(pde_type)+'.csv')
        train_data_bd       = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/train/train_boundary_{beta}_{nu}_{rho}_'+str(pde_type)+'.csv')
        test_data           = pd.read_csv('./data_gen/dataset'+'/'+str(pde_type)+f'/test/test_{beta}_{nu}_{rho}_'+str(pde_type)+'.csv')


    f_dataset = P2INN_Dataset(df=train_data_f)
    u_dataset = P2INN_Dataset(df=train_data_u)
    bd_dataset = P2INN_Dataset_bd(df=train_data_bd)
    test_dataset = P2INN_Dataset(df=test_data)
    
    f_dataloader = DataLoader(f_dataset, batch_size=20000, shuffle=True)
    u_dataloader = DataLoader(u_dataset, batch_size=20000, shuffle=True)
    bd_dataloader = DataLoader(bd_dataset, batch_size=20000, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=20000, shuffle=True)
    
    return f_dataloader, u_dataloader, bd_dataloader, test_dataloader


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)