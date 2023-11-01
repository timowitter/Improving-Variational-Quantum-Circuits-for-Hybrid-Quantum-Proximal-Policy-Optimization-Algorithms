#!/bin/bash

# Specify hyperparameters
# circuits: "no_q_circuit" / "simple" / "simple_reuploading" / "simple_reuploading_with_input_scaleing" /
#          "Hgog" / "Hgog_reuploading" / "Hgog_reuploading_with_input_scaleing" /
#          "Jerbi-no-reuploading-no-input-scaleing" / "Jerbi-reuploading-no-input-scaleing" / "Jerbi-reuploading"

name1a="FL-ppo-ac-NN(3)-(actor-lr=1.0e-3)-(67-params)"
name1b="FL-ppo-ac-NN(3)-(actor-lr=1.0e-2)-(67-params)"
name1c="FL-ppo-ac-NN(3)-(actor-lr=5.0e-2)-(67-params)"

name2a="FL-ppo-ac-NN(4)-(actor-lr=1.0e-3)-(88-params)"
name2b="FL-ppo-ac-NN(4)-(actor-lr=1.0e-2)-(88-params)"
name2c="FL-ppo-ac-NN(4)-(actor-lr=5.0e-2)-(88-params)"

name3a="FL-ppo-ac-NN(5)-(actor-lr=1.0e-3)-(109-params)"
name3b="FL-ppo-ac-NN(5)-(actor-lr=1.0e-2)-(109-params)"
name3c="FL-ppo-ac-NN(5)-(actor-lr=5.0e-2)-(109-params)"

name4a="FL-ppo-ac-NN(4,4)-(actor-lr=1.0e-3)-(108-params)"
name4b="FL-ppo-ac-NN(4,4)-(actor-lr=1.0e-2)-(108-params)"
name4c="FL-ppo-ac-NN(4,4)-(actor-lr=5.0e-2)-(108-params)"

name5a="FL-ppo-ac-NN(64,64)-(actor-lr=5.0e-5)-(5508-params)"
name5b="FL-ppo-ac-NN(64,64)-(actor-lr=2.5e-4)-(5508-params)"
name5c="FL-ppo-ac-NN(64,64)-(actor-lr=1.0e-3)-(5508-params)"

start_seed=10
seed_step=10
end_seed=50
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0") 
timesteps=150000

circuits=("classic_NN_6-6")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name1a-$seed" jobs/job.sh  --exp-name $name1a  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 1.0e-3 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 3 --actor-hidden-layer2-nodes 0 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
            sbatch --job-name="run-$env-$name1b-$seed" jobs/job.sh  --exp-name $name1b  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 1.0e-2 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 3 --actor-hidden-layer2-nodes 0 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
            sbatch --job-name="run-$env-$name1c-$seed" jobs/job.sh  --exp-name $name1c  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 5.0e-2 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 3 --actor-hidden-layer2-nodes 0 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
        done
    done
done





