#!/bin/bash

# Specify hyperparameters
# circuits: "no_q_circuit" / "simple" / "simple_reuploading" / "simple_reuploading_with_input_scaleing" /
#          "Hgog" / "Hgog_reuploading" / "Hgog_reuploading_with_output_scaleing" /
#          "Jerbi-no-reuploading-no-input-scaleing" / "Jerbi-reuploading-no-input-scaleing" / "Jerbi-reuploading"

name1="qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-(72-params)"
name2="qppo-simple_reuploading-qlr0.5e-3-6varlayers-outscale_scheduling_4.0-(72-params)"

start_seed=10
seed_step=10
end_seed=30
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0") 
timesteps=500000

circuits=("simple_reuploading")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name1-$seed" jobs/job.sh --exp-name $name1 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling False --output-scaleing False --load-chkpt True --log-circuit-output False
            sbatch --job-name="run-$env-$name2-$seed" jobs/job.sh --exp-name $name2 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling False --output-scaleing True  --load-chkpt True --scheduled-output-scaleing True --sced-out-scale-fac 4.0
        done
    done
done

name3="qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-clipped-(72-params)"
name4="qppo-simple_reuploading-qlr0.5e-3-6varlayers-outscale_scheduling_2.0-(72-params)"
name5="qppo-simple_reuploading-qlr1.0e-3-6varlayers-nologoutput-clipcoef0.02-(72-params)"
name6="qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-clipcircuitgradnorm-(72-params)"

start_seed=10
seed_step=10
end_seed=30
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0") 
timesteps=200000

circuits=("simple_reuploading")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name3-$seed" jobs/job.sh --exp-name $name3 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling False --output-scaleing False --load-chkpt False --log-circuit-output False --clip-circuit-output True
            sbatch --job-name="run-$env-$name4-$seed" jobs/job.sh --exp-name $name4 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling False --output-scaleing True  --load-chkpt False --scheduled-output-scaleing True --sced-out-scale-fac 2.0
            sbatch --job-name="run-$env-$name5-$seed" jobs/job.sh --exp-name $name5 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 1.0e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling False --output-scaleing False --load-chkpt False --log-circuit-output False --clip-coef 0.02
            sbatch --job-name="run-$env-$name6-$seed" jobs/job.sh --exp-name $name6 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling False --output-scaleing False --load-chkpt False --log-circuit-output False --clip-circuit-grad-norm True
        done
    done
done


#default settings ppo:  'python main.py --exp-name ppo_default  --circuit "simple" --seed 1 --gym-id Deterministic-ShortestPath-4x4-FrozenLake-v0 --num-steps 128 --total-timesteps 300000 --learning-rate 2.5e-4 --qlearning-rate 0e-3 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 4 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False')

#default settings qppo: 'python main.py --exp-name qppo_default --circuit "simple" --seed 1 --gym-id Deterministic-ShortestPath-4x4-FrozenLake-v0 --num-steps 128 --total-timesteps 400000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate 5e-3 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 1 --hybrid False --anneal-lr True --load-chkpt False')



