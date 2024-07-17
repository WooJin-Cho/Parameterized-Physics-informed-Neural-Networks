for seed in 0
do
    for init_cond in 'gauss_pi_2'
    do
        for pde_type in 'checkpoint_all'
        do
            for epoch in 20000
            do
                for coeff_range in 5
                do
                    CUDA_VISIBLE_DEVICES=0 python -u test_all.py --init_cond $init_cond --epoch $epoch --coeff_range $coeff_range --seed $seed --pde_type $pde_type > ./log/test/checkpoint_{$init_cond}_{$pde_type}_{$coeff_range}_{$epoch}_{$seed}.csv
                done
            done
        done
    done
done