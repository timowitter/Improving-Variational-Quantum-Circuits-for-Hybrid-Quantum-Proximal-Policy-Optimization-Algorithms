#!/bin/bash

# Specify hyperparameters
# circuits: "no_q_circuit" / "simple" / "simple_reuploading" / "simple_reuploading_with_input_scaleing" /
#          "Hgog" / "Hgog_reuploading" / "Hgog_reuploading_with_input_scaleing" /
#          "Jerbi-no-reuploading-no-input-scaleing" / "Jerbi-reuploading-no-input-scaleing" / "Jerbi-reuploading"

name1a="CP-ppo-ac-NN(5,5)-(actor-lr=1.0e-4)-(67-params)"
name1b="CP-ppo-ac-NN(5,5)-(actor-lr=1.0e-3)-(67-params)"
name1c="CP-ppo-ac-NN(5,5)-(actor-lr=1.0e-2)-(67-params)"

name2a="CP-ppo-ac-NN(6,5)-(actor-lr=1.0e-4)-(77-params)"
name2b="CP-ppo-ac-NN(6,5)-(actor-lr=1.0e-3)-(77-params)"
name2c="CP-ppo-ac-NN(6,5)-(actor-lr=1.0e-2)-(77-params)"

name3a="CP-ppo-ac-NN(6,6)-(actor-lr=1.0e-4)-(86-params)"
name3b="CP-ppo-ac-NN(6,6)-(actor-lr=1.0e-3)-(86-params)"
name3c="CP-ppo-ac-NN(6,6)-(actor-lr=1.0e-2)-(86-params)"

name4a="CP-ppo-ac-NN(7,7)-(actor-lr=1.0e-4)-(107-params)"
name4b="CP-ppo-ac-NN(7,7)-(actor-lr=1.0e-3)-(107-params)"
name4c="CP-ppo-ac-NN(7,7)-(actor-lr=1.0e-2)-(107-params)"

name5b="CP-ppo-ac-NN(64,64)-(actor-lr=2.5e-4)-(4610-params)"
name5c="CP-ppo-ac-NN(64,64)-(actor-lr=1.0e-3)-(4610-params)"
name5a="CP-ppo-ac-NN(64,64)-(actor-lr=5.0e-3)-(4610-params)"

start_seed=10
seed_step=10
end_seed=50
envs=("CartPole-v1")
timesteps=750000

circuits=("classic_NN")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name5b-$seed" jobs/job.sh  --exp-name $name5b  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 2.5e-4 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 64 --actor-hidden-layer2-nodes 64 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
            sbatch --job-name="run-$env-$name5c-$seed" jobs/job.sh  --exp-name $name5c  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 1.0e-3 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 64 --actor-hidden-layer2-nodes 64 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
            sbatch --job-name="run-$env-$name5a-$seed" jobs/job.sh  --exp-name $name5a  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 5.0e-3 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 64 --actor-hidden-layer2-nodes 64 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
        done
    done
done


