#!/bin/bash

# Specify hyperparameters
# circuits: "no_q_circuit" / "simple" / "simple_reuploading" / "simple_reuploading_with_input_scaleing" /
#          "Hgog" / "Hgog_reuploading" / "Hgog_reuploading_with_input_scaleing" /
#          "Jerbi-no-reuploading-no-input-scaleing" / "Jerbi-reuploading-no-input-scaleing" / "Jerbi-reuploading"



name1f="CP-ppo-ac-NN(5,5)-sigmoid-actorlr((25-5)e-5)-(67-params)"
name2f="CP-ppo-ac-NN(6,5)-sigmoid-actorlr((25-5)e-5)-(77-params)"
name3f="CP-ppo-ac-NN(6,6)-sigmoid-actorlr((25-5)e-5)-(86-params)"
name4f="CP-ppo-ac-NN(7,7)-sigmoid-actorlr((25-5)e-5)-(107-params)"
name5f="CP-ppo-ac-NN(64,64)-sigmoid-actorlr((10-2)e-5)-(4610-params)"


start_seed=10
seed_step=10
end_seed=50
envs=("CartPole-v1") 
timesteps=500000

circuits=("classic_NN")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name1f-$seed" jobs/job.sh  --exp-name $name1f  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 5.0e-5 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 5 --actor-hidden-layer2-nodes 5 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False --exp-qlr-scheduling False --exp-scheduling-halftime 500000 --exp-scheduling-qlearning-rate 2.5e-4

            sbatch --job-name="run-$env-$name2f-$seed" jobs/job.sh  --exp-name $name2f  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 5.0e-5 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 6 --actor-hidden-layer2-nodes 5 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False --exp-qlr-scheduling False --exp-scheduling-halftime 500000 --exp-scheduling-qlearning-rate 2.5e-4

            sbatch --job-name="run-$env-$name3f-$seed" jobs/job.sh  --exp-name $name3f  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 5.0e-5 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 6 --actor-hidden-layer2-nodes 6 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False --exp-qlr-scheduling False --exp-scheduling-halftime 500000 --exp-scheduling-qlearning-rate 2.5e-4

            sbatch --job-name="run-$env-$name4f-$seed" jobs/job.sh  --exp-name $name4f  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 5.0e-5 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 7 --actor-hidden-layer2-nodes 7 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False --exp-qlr-scheduling False --exp-scheduling-halftime 500000 --exp-scheduling-qlearning-rate 2.5e-4

            sbatch --job-name="run-$env-$name5f-$seed" jobs/job.sh  --exp-name $name5f  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 2.0e-5 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 64 --actor-hidden-layer2-nodes 64 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False --exp-qlr-scheduling False --exp-scheduling-halftime 500000 --exp-scheduling-qlearning-rate 1.0e-4
        done
    done
done

