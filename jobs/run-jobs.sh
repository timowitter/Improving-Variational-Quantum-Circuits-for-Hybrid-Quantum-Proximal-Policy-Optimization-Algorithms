#!/bin/bash

# Specify hyperparameters
# circuits: "no_q_circuit" / "simple" / "simple_reuploading" / "simple_reuploading_with_input_scaleing" /
#          "Hgog" / "Hgog_reuploading" / "Hgog_reuploading_with_output_scaleing" /
#          "Jerbi-no-reuploading-no-input-scaleing" / "Jerbi-reuploading-no-input-scaleing" / "Jerbi-reuploading"


name9="FL-qppo-ac-simple_reuploading-exp_sced-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-clippedrandominit-6varlayers-(72-params)"
name10="FL-qppo-ac-simple_reuploading-exp_sced-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-gaussinit-6varlayers-(72-params)"
name11="FL-qppo-ac-simple_reuploading-exp_sced-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-allsmallinit-6varlayers-(72-params)"
name12="FL-qppo-ac-simple_reuploading-exp_sced-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-allmidinit-6varlayers-(72-params)"
name13="FL-qppo-ac-simple_reuploading-exp_sced-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-allbiginit-6varlayers-(72-params)"

start_seed=10
seed_step=10
end_seed=30
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0") 
timesteps=150000

circuits=("simple_reuploading")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name9-$seed" jobs/job.sh  --exp-name $name9  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling True --output-scaleing False --load-chkpt False --exp-scheduling-timesteps 25000 --exp-scheduling-qlearning-rate 10e-3 --param-init random_clipped     --record-grads True
            sbatch --job-name="run-$env-$name10-$seed" jobs/job.sh --exp-name $name10 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling True --output-scaleing False --load-chkpt False --exp-scheduling-timesteps 25000 --exp-scheduling-qlearning-rate 10e-3 --param-init gauss_distribution --record-grads True
            sbatch --job-name="run-$env-$name11-$seed" jobs/job.sh --exp-name $name11 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling True --output-scaleing False --load-chkpt False --exp-scheduling-timesteps 25000 --exp-scheduling-qlearning-rate 10e-3 --param-init allsmall           --record-grads True
            sbatch --job-name="run-$env-$name12-$seed" jobs/job.sh --exp-name $name12 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling True --output-scaleing False --load-chkpt False --exp-scheduling-timesteps 25000 --exp-scheduling-qlearning-rate 10e-3 --param-init allmid             --record-grads True
            sbatch --job-name="run-$env-$name13-$seed" jobs/job.sh --exp-name $name13 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --exp-qlr-scheduling True --output-scaleing False --load-chkpt False --exp-scheduling-timesteps 25000 --exp-scheduling-qlearning-rate 10e-3 --param-init allbig             --record-grads True
        done
    done
done





name7="FL-qppo-ac-simple_reuploading-qlr2.5e-3-lr2.5e-4-2varlayers-(24-params)"
name8="FL-qppo-ac-simple_reuploading-qlr2.5e-3-lr2.5e-4-4varlayers-(48-params)"

start_seed=10
seed_step=10
end_seed=30
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0") 
timesteps=150000

circuits=("simple_reuploading")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name7-$seed" jobs/job.sh --exp-name $name7 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 2.5e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 2  --output-scaleing False --load-chkpt False
            sbatch --job-name="run-$env-$name8-$seed" jobs/job.sh --exp-name $name8 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 2.5e-3 --n-qubits 4 --n-var-layers 4 --n-enc-layers 4  --output-scaleing False --load-chkpt False
        done
    done
done



name1="FL-qppo-ac-simple_reuploading-exp_sced-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-2varlayers-(24-params)"

start_seed=10
seed_step=10
end_seed=10
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0") 
timesteps=150000

circuits=("simple_reuploading")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name1-$seed" jobs/job.sh --exp-name $name1 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 2 --exp-qlr-scheduling True --output-scaleing False --load-chkpt False --exp-scheduling-timesteps 25000 --exp-scheduling-qlearning-rate 10e-3
        done
    done
done


#--weight-remapping none / clipped / pos_clipped / tanh / double_tanh / pos_tanh
#default settings ppo:  'python main.py --exp-name ppo_default  --circuit "simple" --seed 1 --gym-id Deterministic-ShortestPath-4x4-FrozenLake-v0 --num-steps 128 --total-timesteps 300000 --learning-rate 2.5e-4 --qlearning-rate 0e-3 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 4 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False')

#default settings qppo: 'python main.py --exp-name qppo_default --circuit "simple" --seed 1 --gym-id Deterministic-ShortestPath-4x4-FrozenLake-v0 --num-steps 128 --total-timesteps 400000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate 5e-3 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 1 --hybrid False --anneal-lr True --load-chkpt False')



