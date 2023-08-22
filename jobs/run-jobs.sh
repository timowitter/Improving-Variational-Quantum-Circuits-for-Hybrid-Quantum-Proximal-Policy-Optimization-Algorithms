#!/bin/bash

# Specify hyperparameters
# circuits: "no_q_circuit" / "simple" / "simple_reuploading" / "simple_reuploading_with_input_scaleing" /
#          "Hgog" / "Hgog_reuploading" / "Hgog_reuploading_with_output_scaleing" /
#          "Jerbi-no-reuploading-no-input-scaleing" / "Jerbi-reuploading-no-input-scaleing" / "Jerbi-reuploading"


name1="FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)"
name16="FL-qppo-simple_reuploading-exp_sced-start-qlr10.e-3-end10e4-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)"
name19="FL-qppo-simple_reuploading-double_sced-start-qlr10.e-3-mid05e4-qlr2.5e-3-end15e4-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)"

start_seed=40
seed_step=10
end_seed=50
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0") 
timesteps=300000

circuits=("simple_reuploading")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name1-$seed" jobs/job.sh --exp-name $name1 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 2.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling False --output-scaleing False --load-chkpt False
            sbatch --job-name="run-$env-$name16-$seed" jobs/job.sh --exp-name $name16 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling True --output-scaleing False --load-chkpt False --clip-coef 0.2 --exp-scheduling-timesteps 100000 --exp-scheduling-qlearning-rate 10e-3
            sbatch --job-name="run-$env-$name19-$seed" jobs/job.sh --exp-name $name19 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling True --output-scaleing False --load-chkpt False --clip-coef 0.2 --exp-scheduling-timesteps  50000 --exp-scheduling-qlearning-rate 10e-3 --lin-qlr-scheduling True --lin-scheduling-timesteps 150000 --lin-scheduling-qlearning-rate 2.5e-3
        done
    done
done


start_seed=10
seed_step=10
end_seed=30
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0") 
timesteps=300000

circuits=("simple_reuploading")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name1-$seed" jobs/job.sh --exp-name $name1 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 2.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling False --output-scaleing False --load-chkpt True
            sbatch --job-name="run-$env-$name16-$seed" jobs/job.sh --exp-name $name16 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling True --output-scaleing False --load-chkpt True --clip-coef 0.2 --exp-scheduling-timesteps 100000 --exp-scheduling-qlearning-rate 10e-3
            sbatch --job-name="run-$env-$name19-$seed" jobs/job.sh --exp-name $name19 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling True --output-scaleing False --load-chkpt True --clip-coef 0.2 --exp-scheduling-timesteps  50000 --exp-scheduling-qlearning-rate 10e-3 --lin-qlr-scheduling True --lin-scheduling-timesteps 150000 --lin-scheduling-qlearning-rate 2.5e-3
        done
    done
done

#--weight-remapping none / clipped / pos_clipped / tanh / double_tanh / pos_tanh
#default settings ppo:  'python main.py --exp-name ppo_default  --circuit "simple" --seed 1 --gym-id Deterministic-ShortestPath-4x4-FrozenLake-v0 --num-steps 128 --total-timesteps 300000 --learning-rate 2.5e-4 --qlearning-rate 0e-3 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 4 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False')

#default settings qppo: 'python main.py --exp-name qppo_default --circuit "simple" --seed 1 --gym-id Deterministic-ShortestPath-4x4-FrozenLake-v0 --num-steps 128 --total-timesteps 400000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate 5e-3 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 1 --hybrid False --anneal-lr True --load-chkpt False')



