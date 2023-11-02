#!/bin/bash

# Specify hyperparameters
# circuits: "no_q_circuit" / "simple" / "simple_reuploading" / "simple_reuploading_with_input_scaleing" /
#          "Hgog" / "Hgog_reuploading" / "Hgog_reuploading_with_input_scaleing" /
#          "Jerbi-no-reuploading-no-input-scaleing" / "Jerbi-reuploading-no-input-scaleing" / "Jerbi-reuploading"


name1="FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-5e-2-(73-params)"
name9="FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-1e-2-(73-params)-randominit"
name10="FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-1e-2-(73-params)-gaussinit"
name11="FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-1e-2-(73-params)-allsmallinit"
name12="FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-1e-2-(73-params)-allmidinit"
name13="FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-1e-2-(73-params)-allbiginit"

start_seed=10
seed_step=10
end_seed=10
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0") 
timesteps=50000

circuits=("simple_reuploading")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name1-$seed" jobs/job.sh  --exp-name $name1  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt False --exp-qlr-scheduling True --exp-scheduling-halftime 25000 --exp-scheduling-qlearning-rate 10e-3 --output-scaleing True --output-scaleing-learning-rate 5e-2 --shared-output-scaleing-param True
            #sbatch --job-name="run-$env-$name9-$seed" jobs/job.sh  --exp-name $name9  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt False --exp-qlr-scheduling True --exp-scheduling-halftime 25000 --exp-scheduling-qlearning-rate 10e-3 --output-scaleing True --output-scaleing-learning-rate 1e-2 --shared-output-scaleing-param True --param-init fully_random  --record-grads True
            #sbatch --job-name="run-$env-$name10-$seed" jobs/job.sh --exp-name $name10 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling True --load-chkpt False --exp-scheduling-halftime 25000 --exp-scheduling-qlearning-rate 10e-3 --output-scaleing True --output-scaleing-learning-rate 1e-2 --shared-output-scaleing-param True --param-init unclipped_gauss_distribution  --record-grads True
            #sbatch --job-name="run-$env-$name11-$seed" jobs/job.sh --exp-name $name11 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling True --load-chkpt False --exp-scheduling-halftime 25000 --exp-scheduling-qlearning-rate 10e-3 --output-scaleing True --output-scaleing-learning-rate 1e-2 --shared-output-scaleing-param True --param-init allsmall        --record-grads True
            #sbatch --job-name="run-$env-$name12-$seed" jobs/job.sh --exp-name $name12 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling True --load-chkpt False --exp-scheduling-halftime 25000 --exp-scheduling-qlearning-rate 10e-3 --output-scaleing True --output-scaleing-learning-rate 1e-2 --shared-output-scaleing-param True --param-init allmid          --record-grads True
            #sbatch --job-name="run-$env-$name13-$seed" jobs/job.sh --exp-name $name13 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling True --load-chkpt False --exp-scheduling-halftime 25000 --exp-scheduling-qlearning-rate 10e-3 --output-scaleing True --output-scaleing-learning-rate 1e-2 --shared-output-scaleing-param True --param-init allbig          --record-grads True
        done
    done
done


start_seed=30
seed_step=10
end_seed=30
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0") 
timesteps=50000

circuits=("simple_reuploading")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            #sbatch --job-name="run-$env-$name1-$seed" jobs/job.sh  --exp-name $name1  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt False --exp-qlr-scheduling True --exp-scheduling-halftime 25000 --exp-scheduling-qlearning-rate 10e-3 --output-scaleing True --output-scaleing-learning-rate 5e-2 --shared-output-scaleing-param True
            #sbatch --job-name="run-$env-$name9-$seed" jobs/job.sh  --exp-name $name9  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt False --exp-qlr-scheduling True --exp-scheduling-halftime 25000 --exp-scheduling-qlearning-rate 10e-3 --output-scaleing True --output-scaleing-learning-rate 1e-2 --shared-output-scaleing-param True --param-init fully_random  --record-grads True
            sbatch --job-name="run-$env-$name10-$seed" jobs/job.sh --exp-name $name10 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling True --load-chkpt False --exp-scheduling-halftime 25000 --exp-scheduling-qlearning-rate 10e-3 --output-scaleing True --output-scaleing-learning-rate 1e-2 --shared-output-scaleing-param True --param-init unclipped_gauss_distribution  --record-grads True
            sbatch --job-name="run-$env-$name11-$seed" jobs/job.sh --exp-name $name11 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling True --load-chkpt False --exp-scheduling-halftime 25000 --exp-scheduling-qlearning-rate 10e-3 --output-scaleing True --output-scaleing-learning-rate 1e-2 --shared-output-scaleing-param True --param-init allsmall        --record-grads True
            sbatch --job-name="run-$env-$name12-$seed" jobs/job.sh --exp-name $name12 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling True --load-chkpt False --exp-scheduling-halftime 25000 --exp-scheduling-qlearning-rate 10e-3 --output-scaleing True --output-scaleing-learning-rate 1e-2 --shared-output-scaleing-param True --param-init allmid          --record-grads True
            sbatch --job-name="run-$env-$name13-$seed" jobs/job.sh --exp-name $name13 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling True --load-chkpt False --exp-scheduling-halftime 25000 --exp-scheduling-qlearning-rate 10e-3 --output-scaleing True --output-scaleing-learning-rate 1e-2 --shared-output-scaleing-param True --param-init allbig          --record-grads True
        done
    done
done