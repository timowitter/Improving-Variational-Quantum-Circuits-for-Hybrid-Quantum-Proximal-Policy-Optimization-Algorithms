#!/bin/bash

# Specify hyperparameters
# circuits: "no_q_circuit" / "simple" / "simple_reuploading" / "simple_reuploading_with_input_scaleing" /
#          "Hgog" / "Hgog_reuploading" / "Hgog_reuploading_with_output_scaleing" /
#          "Jerbi-no-reuploading-no-input-scaleing" / "Jerbi-reuploading-no-input-scaleing" / "Jerbi-reuploading"



name7="CP-ppo-ac-NN(5,5)-(lr=2.5e-3)-(67-params)"
name8="CP-ppo-ac-NN(6,6)-(lr=2.5e-3)-(86-params)"
name9="CP-ppo-ac-NN(7,7)-(lr=2.5e-3)-(107-params)"
name10="CP-ppo-ac-NN(5,5)-(lr=1.0e-3)-(67-params)"
name11="CP-ppo-ac-NN(7,7)-(lr=1.0e-3)-(107-params)"


start_seed=10
seed_step=10
end_seed=50
envs=("CartPole-v1") 
timesteps=500000

circuits=("classic_NN_6-6")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name7-$seed" jobs/job.sh  --exp-name $name7  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-3 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 5 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
            sbatch --job-name="run-$env-$name8-$seed" jobs/job.sh  --exp-name $name8  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-3 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 6 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
            sbatch --job-name="run-$env-$name9-$seed" jobs/job.sh  --exp-name $name9  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-3 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 7 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
            sbatch --job-name="run-$env-$name10-$seed" jobs/job.sh --exp-name $name10 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 1.0e-3 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 5 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
            sbatch --job-name="run-$env-$name11-$seed" jobs/job.sh --exp-name $name11 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 1.0e-3 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 7 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
        done
    done
done


name00="CP-random-baseline-(0-params)"

start_seed=10
seed_step=10
end_seed=50
envs=("CartPole-v1") 
timesteps=500000

circuits=("random-baseline")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name00-$seed" jobs/job.sh --exp-name $name00 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 0      --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 2 --critic-hidden-layer-nodes 2  --quantum-actor False --load-chkpt False --random-baseline True
        done
    done
done



#--weight-remapping none / clipped / pos_clipped / tanh / double_tanh / pos_tanh
#default settings ppo:  'python main.py --exp-name ppo_default  --circuit "simple" --seed 1 --gym-id Deterministic-ShortestPath-4x4-FrozenLake-v0 --num-steps 128 --total-timesteps 300000 --learning-rate 2.5e-4 --qlearning-rate 0e-3 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 4 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False')

#default settings qppo: 'python main.py --exp-name qppo_default --circuit "simple" --seed 1 --gym-id Deterministic-ShortestPath-4x4-FrozenLake-v0 --num-steps 128 --total-timesteps 400000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate 5e-3 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 1 --hybrid False --anneal-lr True --load-chkpt False')



