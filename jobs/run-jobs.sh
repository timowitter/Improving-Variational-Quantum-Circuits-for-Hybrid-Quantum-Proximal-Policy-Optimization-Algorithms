#!/bin/bash

# Specify hyperparameters
# circuits: "no_q_circuit" / "simple" / "simple_reuploading" / "simple_reuploading_with_input_scaleing" /
#          "Hgog" / "Hgog_reuploading" / "Hgog_reuploading_with_output_scaleing" /
#          "Jerbi-no-reuploading-no-input-scaleing" / "Jerbi-reuploading-no-input-scaleing" / "Jerbi-reuploading"


name1="FL-qppo-ac-simple_reuploading-exp_sced-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-2varlayers-(24-params)"
name2="FL-qppo-ac-simple_reuploading-exp_sced-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-4varlayers-(48-params)"
name3="FL-qppo-simple_reuploading-exp_sced_fixed-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-6varlayers-(72-params)",
name4="FL-qppo-ac-simple_reuploading-exp_sced-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-8varlayers-(96-params)"
name5="FL-qppo-ac-simple_reuploading-exp_sced-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-10varlayers-(120-params)"
name6="FL-qppo-ac-simple_reuploading-qlr2.5e-3-lr2.5e-4-8varlayers-(96-params)"

start_seed=10
seed_step=10
end_seed=30
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0") 
timesteps=150000

circuits=("simple_reuploading")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name1-$seed" jobs/job.sh --exp-name $name1 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 2 --exp-qlr-scheduling True --output-scaleing False --load-chkpt False --exp-scheduling-timesteps 25000 --exp-scheduling-qlearning-rate 10e-3
            sbatch --job-name="run-$env-$name2-$seed" jobs/job.sh --exp-name $name2 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 4 --n-enc-layers 4 --exp-qlr-scheduling True --output-scaleing False --load-chkpt False --exp-scheduling-timesteps 25000 --exp-scheduling-qlearning-rate 10e-3
            sbatch --job-name="run-$env-$name4-$seed" jobs/job.sh --exp-name $name4 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 8 --n-enc-layers 8 --exp-qlr-scheduling True --output-scaleing False --load-chkpt False --exp-scheduling-timesteps 25000 --exp-scheduling-qlearning-rate 10e-3
            sbatch --job-name="run-$env-$name5-$seed" jobs/job.sh --exp-name $name5 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 10 --n-enc-layers 10 --exp-qlr-scheduling True --output-scaleing False --load-chkpt False --exp-scheduling-timesteps 25000 --exp-scheduling-qlearning-rate 10e-3
            sbatch --job-name="run-$env-$name6-$seed" jobs/job.sh --exp-name $name6 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 2.5e-3 --n-qubits 4 --n-var-layers 8 --n-enc-layers 8  --output-scaleing False --load-chkpt False
        done
    done
done

#--weight-remapping none / clipped / pos_clipped / tanh / double_tanh / pos_tanh
#default settings ppo:  'python main.py --exp-name ppo_default  --circuit "simple" --seed 1 --gym-id Deterministic-ShortestPath-4x4-FrozenLake-v0 --num-steps 128 --total-timesteps 300000 --learning-rate 2.5e-4 --qlearning-rate 0e-3 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 4 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False')

#default settings qppo: 'python main.py --exp-name qppo_default --circuit "simple" --seed 1 --gym-id Deterministic-ShortestPath-4x4-FrozenLake-v0 --num-steps 128 --total-timesteps 400000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate 5e-3 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 1 --hybrid False --anneal-lr True --load-chkpt False')



