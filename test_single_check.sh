for seed in 0
do
    for init_cond in 'gauss_pi_2'
    do
        for pde_type in 'checkpoint_reaction'
            do
            for epoch in 10000
            do
                for coeff_range in 20
                do
                    python -u test_single.py --init_cond $init_cond --epoch $epoch --coeff_range $coeff_range --pde_type $pde_type --seed $seed > ./log/test/checkpoint_{$init_cond}_{$pde_type}_{$coeff_range}_{$epoch}_{$seed}.csv
                done
            done
        done
    done
done