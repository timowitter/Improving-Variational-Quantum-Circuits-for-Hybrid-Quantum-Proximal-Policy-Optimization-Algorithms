#!/bin/bash

# Specify hyperparameters
# circuits: "no_q_circuit" / "simple" / "simple_reuploading" / "simple_reuploading_with_input_scaleing" /
#          "Hgog" / "Hgog_reuploading" / "Hgog_reuploading_with_input_scaleing" /
#          "Jerbi-no-reuploading-no-input-scaleing" / "Jerbi-reuploading-no-input-scaleing" / "Jerbi-reuploading"

start_seed=10
seed_step=10
end_seed=50
envs=("CartPole-v1")
timesteps=750000

circuits=("classic_NN")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name3a-$seed" jobs/job.sh  --exp-name $name3a  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 1.0e-4 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 6 --actor-hidden-layer2-nodes 6 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
            sbatch --job-name="run-$env-$name3b-$seed" jobs/job.sh  --exp-name $name3b  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 1.0e-3 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 6 --actor-hidden-layer2-nodes 6 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
            sbatch --job-name="run-$env-$name3c-$seed" jobs/job.sh  --exp-name $name3c  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 1.0e-2 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 6 --actor-hidden-layer2-nodes 6 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False

            sbatch --job-name="run-$env-$name4a-$seed" jobs/job.sh  --exp-name $name4a  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 1.0e-4 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 7 --actor-hidden-layer2-nodes 7 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
            sbatch --job-name="run-$env-$name4b-$seed" jobs/job.sh  --exp-name $name4b  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 1.0e-3 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 7 --actor-hidden-layer2-nodes 7 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
            sbatch --job-name="run-$env-$name4c-$seed" jobs/job.sh  --exp-name $name4c  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 1.0e-2 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 7 --actor-hidden-layer2-nodes 7 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
        done
    done
done
