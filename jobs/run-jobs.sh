#!/bin/bash

# Specify hyperparameters
# circuits: "no_q_circuit" / "simple" / "simple_reuploading" / "simple_reuploading_with_input_scaleing" /
#          "Hgog" / "Hgog_reuploading" / "Hgog_reuploading_with_output_scaleing" /
#          "Jerbi-no-reuploading-no-input-scaleing" / "Jerbi-reuploading-no-input-scaleing" / "Jerbi-reuploading"



name20="FL-qppo-simple_reuploading-exp_sced-start-qlr10.e-3-end50e3-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)"
name21="FL-qppo-simple_reuploading-exp_sced-start-qlr10.e-3-end25e3-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)"
name22="FL-qppo-simple_reuploading-double_sced-start-qlr10.e-3-mid25e3-qlr2.5e-3-end15e4-qlr0.5e-3-lr0.5e-4-6varlayers-(72-params)"
name23="FL-qppo-simple_reuploading-double_sced-start-qlr10.e-3-mid25e3-qlr2.5e-3-end15e4-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)"
name24="FL-qppo-simple_reuploading-double_sced-start-qlr10.e-3-mid25e3-qlr2.5e-3-end15e4-qlr0.5e-3-lr10.e-4-6varlayers-(72-params)"

start_seed=10
seed_step=10
end_seed=30
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0") 
timesteps=150000

circuits=("simple_reuploading")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name20-$seed" jobs/job.sh --exp-name $name20 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling True --output-scaleing False --load-chkpt False --clip-coef 0.2 --exp-scheduling-timesteps 50000 --exp-scheduling-qlearning-rate 10e-3
            sbatch --job-name="run-$env-$name21-$seed" jobs/job.sh --exp-name $name21 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling True --output-scaleing False --load-chkpt False --clip-coef 0.2 --exp-scheduling-timesteps 25000 --exp-scheduling-qlearning-rate 10e-3
            sbatch --job-name="run-$env-$name22-$seed" jobs/job.sh --exp-name $name22 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 0.5e-4 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling True --output-scaleing False --load-chkpt False --clip-coef 0.2 --exp-scheduling-timesteps 25000 --exp-scheduling-qlearning-rate 10e-3 --lin-qlr-scheduling True --lin-scheduling-timesteps 150000 --lin-scheduling-qlearning-rate 2.5e-3
            sbatch --job-name="run-$env-$name23-$seed" jobs/job.sh --exp-name $name23 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling True --output-scaleing False --load-chkpt False --clip-coef 0.2 --exp-scheduling-timesteps 25000 --exp-scheduling-qlearning-rate 10e-3 --lin-qlr-scheduling True --lin-scheduling-timesteps 150000 --lin-scheduling-qlearning-rate 2.5e-3
            sbatch --job-name="run-$env-$name24-$seed" jobs/job.sh --exp-name $name24 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 10.0e-4 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling True --output-scaleing False --load-chkpt False --clip-coef 0.2 --exp-scheduling-timesteps 25000 --exp-scheduling-qlearning-rate 10e-3 --lin-qlr-scheduling True --lin-scheduling-timesteps 150000 --lin-scheduling-qlearning-rate 2.5e-3
        done
    done
done




#--weight-remapping none / clipped / pos_clipped / tanh / double_tanh / pos_tanh
#default settings ppo:  'python main.py --exp-name ppo_default  --circuit "simple" --seed 1 --gym-id Deterministic-ShortestPath-4x4-FrozenLake-v0 --num-steps 128 --total-timesteps 300000 --learning-rate 2.5e-4 --qlearning-rate 0e-3 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 4 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False')

#default settings qppo: 'python main.py --exp-name qppo_default --circuit "simple" --seed 1 --gym-id Deterministic-ShortestPath-4x4-FrozenLake-v0 --num-steps 128 --total-timesteps 400000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate 5e-3 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 1 --hybrid False --anneal-lr True --load-chkpt False')



