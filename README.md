# Parameterized Physics-informed Neural Networks for Parameterized PDEs, (ICML 2024, Oral)
## 0. Experimental environment settings.

Run the following code before starting the experiment.

    conda env create -f env.yaml
    conda activate P2INN


## 1. Data generation.

You can generate dataset for train / validation / test. 
Run code in the folder "data_gen".

    
         [ Code ]                   [ Description of code ]

    python gen_conv.py : Code for generating convection equation data
    python gen_diff.py : Code for generating diffusion equation data
    python gen_reac.py : Code for generating reaction equatinon data
    python gen_cd.py   : Code for generating Convection-Diffusion equation data
    python gen_rd.py   : Code for generating Reaction-Diffusion equation data
    python gen_cdr.py  : Code for generating Convection-Diffusion-Reaction data

Set the initial condition using "u0_str" parser. 
(you can select following option : 1+sin(x), gauss, gauss_pi_2, etc...)

    [ u0_str ]

    1+sin(x)    : 1+sin(x)
    gauss       : Gaussian distribution with STD=pi/4.
    gauss_pi_2  : Gaussian distribution with STD=pi/2. (Default)


## 2. Train

Run the following code for P^2INN training.

         [ Code ]              [ Description of code ]

    sh train_all.sh     : Code for P^2INN training (All type)
    sh train_single.sh  : Code for P^2INN training (Single type)

Detailed settings can be changed in config.py


## 3. Test

Run the following code for test.

         [ Code ]              [ Description of code ]

    sh test_all.sh      : Code for testing P^2INN (All type) performance
    sh test_single.sh   : Code for testing P^2INN (Single type) performance

In additaon, we attach the checkpoint of P^2INN (.pt file)
If you want to check it quickly, run the following code below.

         [ Code ]              [ Description of code ]

    sh test_all_check.sh    : Code for testing P^2INN (All type),   (1~5 range)
    sh test_single_check.sh : Code for testing P^2INN (Single type),(1~5 range) 

## 4. Other code

Brief description of the other code files.

         [ Code ]              [ Description of code ]
         
    model.py            :   P^2INN model.
    Loss_f.py           :   PDE residual loss.
    dataloader.py       :   Dataloader used in train / test
    train_svd_mod.py    :   Our proposed SVD modulation method.
