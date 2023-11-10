#!/bin/bash

# Specify hyperparameters
# circuits: "no_q_circuit" / "simple" / "simple_reuploading" / "simple_reuploading_with_input_scaleing" /
#          "Hgog" / "Hgog_reuploading" / "Hgog_reuploading_with_input_scaleing" /
#          "Jerbi-no-reuploading-no-input-scaleing" / "Jerbi-reuploading-no-input-scaleing" / "Jerbi-reuploading"

name1f="CP-ppo-ac-NN(5,5)-actorlr((100-1)e-5,ht=80000)-(67-params)"
name2f="CP-ppo-ac-NN(6,5)-actorlr((100-1)e-5,ht=80000)-(77-params)"
name3f="CP-ppo-ac-NN(6,6)-actorlr((100-1)e-5,ht=80000)-(86-params)"
name4f="CP-ppo-ac-NN(7,7)-actorlr((100-1)e-5,ht=80000)-(107-params)"
name5f="CP-ppo-ac-NN(64,64)-actorlr((25-1)e-5,ht=100000)-(4610-params)"


start_seed=10
seed_step=10
end_seed=50
envs=("CartPole-v1") 
timesteps=500000

circuits=("classic_NN")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name1f-$seed" jobs/job.sh  --exp-name $name1f  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 1.0e-5 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 5 --actor-hidden-layer2-nodes 5 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False --exp-qlr-scheduling True --exp-scheduling-halftime 80000 --exp-scheduling-qlearning-rate 1.0e-3

            sbatch --job-name="run-$env-$name2f-$seed" jobs/job.sh  --exp-name $name2f  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 1.0e-5 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 6 --actor-hidden-layer2-nodes 5 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False --exp-qlr-scheduling True --exp-scheduling-halftime 80000 --exp-scheduling-qlearning-rate 1.0e-3

            sbatch --job-name="run-$env-$name3f-$seed" jobs/job.sh  --exp-name $name3f  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 1.0e-5 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 6 --actor-hidden-layer2-nodes 6 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False --exp-qlr-scheduling True --exp-scheduling-halftime 80000 --exp-scheduling-qlearning-rate 1.0e-3

            sbatch --job-name="run-$env-$name4f-$seed" jobs/job.sh  --exp-name $name4f  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 1.0e-5 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 7 --actor-hidden-layer2-nodes 7 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False --exp-qlr-scheduling True --exp-scheduling-halftime 80000 --exp-scheduling-qlearning-rate 1.0e-3

            sbatch --job-name="run-$env-$name5f-$seed" jobs/job.sh  --exp-name $name5f  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 1.0e-5 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 64 --actor-hidden-layer2-nodes 64 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False --exp-qlr-scheduling True --exp-scheduling-halftime 100000 --exp-scheduling-qlearning-rate 2.5e-4
        done
    done
done
