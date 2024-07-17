for init_cond in 'gauss_pi_2'
do
    for load_epoch in 20000
    do
        for pde_type in 'convection'
        do
            for beta in 3
            do
                for nu in 0
                do
                    for rho in 0
                    do
                        for load_range in 5
                        do
                            CUDA_VISIBLE_DEVICES=0 python -u train_svd_mod.py --init_cond $init_cond --load_epoch $load_epoch --pde_type $pde_type --beta $beta --nu $nu --rho $rho --load_range $load_range > ./log/train/mod_svd_{$init_cond}_{$load_epoch}_{$pde_type}_{$beta}_{$nu}_{$rho}_{$load_range}.csv
                        done
                    done
                done
            done
        done
    done
done
