#ppo based on https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo.py
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import gym
import time
from torch.utils.tensorboard import SummaryWriter

from env_setup import make_env #, save_envs, load_envs
from circuits import actor_circuit_selection, critic_circuit_selection
from agent import Agent
from layer_params import make_actor_layer_params, make_critic_layer_params
from calc_num_params import calc_num_actor_params, calc_num_critic_params
import save_params
from envs_storage import Store_envs
from args import parse_args, save_args

args = parse_args()
pathExists = os.path.exists(args.chkpt_dir)
if not pathExists:
    os.makedirs(args.chkpt_dir)

save_args(args)

if __name__ == "__main__":
    #initialise epsylon
    epsylon=1   #100% random until first update
    
    actor_par_count = calc_num_actor_params()
    critic_par_count = calc_num_critic_params()
    print("calculated number of actor parameters: ",actor_par_count)
    print("calculated number of critic parameters: ",critic_par_count)

    run_name = f"{args.gym_id}__{args.exp_name}__{actor_par_count}__{critic_par_count}__{args.seed}"    #__{int(time.time())}

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic


    # Setup env
    store_envs = Store_envs()
    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name, args.num_envs, args.chkpt_dir, args.load_chkpt, store_envs) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "onely discrete action spaces supported"
    
    
    #Declare Quantum Circuit Parameters and agent
    agent = Agent(envs) 
    actor_layer_params = make_actor_layer_params()                                                  #alternative: layer_params = np.random.normal(0, np.tan(0.5), (args.n_qubits, (2 * args.n_var_layers) + 2, 2))
    output_scaleing_params = np.ones(envs.single_action_space.n)*args.output_scaleing_start
    critic_layer_params = make_critic_layer_params()

    if (args.load_chkpt):                       #load Parameters from checkpoint
        if (args.quantum_actor==False or args.quantum_critic==False):
            agent.load_classical_network_params()
        if (args.quantum_actor):
            actor_layer_params = save_params.load_actor_circuit_params(args.chkpt_dir)
        if (args.quantum_actor and args.hybrid==False and args.epsylon_greedy==False):
            output_scaleing_params = save_params.load_output_scaleing_params(args.chkpt_dir)
        if (args.quantum_critic):
            critic_layer_params = save_params.load_critic_circuit_params(args.chkpt_dir)
    else:                                       #make existing Parameters trainable
        actor_layer_params = Variable(torch.tensor(actor_layer_params), requires_grad=True) 
        output_scaleing_params = Variable(torch.tensor(output_scaleing_params), requires_grad=True)
        critic_layer_params = Variable(torch.tensor(critic_layer_params), requires_grad=True)


    actor_circuit = actor_circuit_selection()
    critic_circuit = critic_circuit_selection()

    optimizer1 = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    if (args.quantum_actor):                                                                            #3 Different Adams epsylon
        optimizer2 = optim.Adam([actor_layer_params], lr=args.qlearning_rate, eps=1e-5)
    if (args.quantum_actor and args.hybrid==False and args.epsylon_greedy==False):                                                                            
        optimizer3 = optim.Adam([output_scaleing_params], lr=args.output_scaleing_learning_rate, eps=1e-5)
    if (args.quantum_critic):
        optimizer4 = optim.Adam([critic_layer_params], lr=args.qlearning_rate, eps=1e-5)

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs)+envs.single_observation_space.shape) 
    actions = torch.zeros((args.num_steps, args.num_envs)+envs.single_action_space.shape, dtype=torch.short) 
    logprobs = torch.zeros((args.num_steps, args.num_envs)) 
    rewards = torch.zeros((args.num_steps, args.num_envs)) 
    dones = torch.zeros((args.num_steps, args.num_envs)) 
    values = torch.zeros((args.num_steps, args.num_envs)) 


    # Game start
    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size
    warmup_updates = args.warmup_timesteps // args.batch_size
    #print(f"Number of Updates planned: {num_updates}")

    if (args.load_chkpt):
        global_step, next_obs, next_done = save_params.load_state(args.chkpt_dir)
        #updates_remaining = (args.total_timesteps - global_step) // args.batch_size
        done_updates = (global_step) // args.batch_size
    else:
        global_step = 0
        next_obs = torch.Tensor(envs.reset()) 
        next_done = torch.zeros(args.num_envs) 
        done_updates = 0

    # Training loop
    for update in range (done_updates + 1, num_updates + 1):
        # Annealing the rate if instructed to do so
        if args.anneal_lr:
            frac_lin = 1.0 - (update - 1.0) / warmup_updates #1 at start, linearly decreasing over time -> lr will decrease over time
            frac_exp = (warmup_updates - update + 1.0)**2 / warmup_updates**2 #1 at start, exponentially decreasing over time -> greedyness will decrease over time
            lrnow1 = frac_lin * args.warmup_learning_rate_bonus + args.learning_rate 
            lrnow2 = frac_exp * args.warmup_qlearning_rate_bonus + args.qlearning_rate
            optimizer1.param_groups[0]["lr"] = lrnow1
            if (args.quantum_actor):
                optimizer2.param_groups[0]["lr"] = lrnow2
            if (args.quantum_critic):
                optimizer4.param_groups[0]["lr"] = lrnow2
        if (args.epsylon_greedy):
            #frac3 = 1.0 - (update - 1.0) / num_updates
            #frac3 = (num_updates-update)**2 / num_updates**2 #1 at start, exponentially decreasing over time -> greedyness will decrease over time
            #epsylon = frac3 * args.epsylon
            epsylon = 0.9999**(global_step)

        # Environment interaction
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step]=next_obs
            dones[step]=next_done

            # Get action
            for i in range(args.num_envs):
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs[i], actor_circuit, actor_layer_params, critic_circuit, critic_layer_params, output_scaleing_params, epsylon, envs.single_observation_space.n, envs.single_action_space.n)
                    values[step, i]= value.flatten()
                actions[step, i]=action.short()
                logprobs[step, i]=logprob

            # Env step
            next_ob, reward, done, info = envs.step(actions[step].cpu().numpy())        #np.asarray(x, dtype = 'int')
            rewards[step] = torch.tensor(reward) .view(-1)
            next_obs, next_done = torch.Tensor(next_ob) , torch.Tensor(done) 
            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic return {item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item['episode']['r'], global_step)
                    writer.add_scalar("charts/episodic_length", item['episode']['l'], global_step)
                    break

        # calculate commulative discounted Returns and Advantages
        with torch.no_grad():
            next_value_i = torch.zeros(args.num_envs) 
            for i in range(args.num_envs):
                next_value_i[i] = agent.get_value(next_obs[i], envs.single_observation_space.n, critic_circuit, critic_layer_params)
            next_value = next_value_i.reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards) 
                nextadvantage = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalue = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalue = values[t + 1]
                    delta = rewards[t] + args.gamma * nextnonterminal * nextvalue - values[t]  #analog zu returns[t] - values[t] nur mit nextvalue anstatt next_return
                    advantages[t] = nextadvantage = delta + args.gamma * args.gae_lambda * nextnonterminal * nextadvantage  # + commulative discounted advantage
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards) 
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return   #commulative discounted return
                advantages = returns - values


        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Start optimizing the policy and value network
        b_inds = np.arange(args.batch_size)         # Make batches
        clipfracs = []
        for epoch in range (args.update_epochs):
            np.random.shuffle(b_inds)               # Shuffle batches
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]         # Make minibatches
                # Start training,
                newlogprob = torch.zeros(args.minibatch_size) 
                entropy = torch.zeros(args.minibatch_size) 
                newvalue = torch.zeros(args.minibatch_size) 

                for i in range(args.minibatch_size):         # Make minibatches        #problem 4 size
                    # Get values, ratio               
                    _, newlogprob_i, entropy_i, newvalue_i = agent.get_action_and_value(b_obs[mb_inds[i]], actor_circuit, actor_layer_params, critic_circuit, critic_layer_params, output_scaleing_params, epsylon, envs.single_observation_space.n, envs.single_action_space.n, b_actions.long()[mb_inds[i]])
                    
                    newlogprob[i] = newlogprob_i
                    entropy[i] = entropy_i
                    newvalue[i] = newvalue_i

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Advantage normalisation
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # debug variables (info)
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()   # the approximate Kullback–Leibler divergence: how agressice does the policy update
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]   # the fraction of the training data that triggered the clipped objective

                # Start loss calcualtion
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                # Overall loss
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
                
                optimizer1.zero_grad()
                if (args.quantum_actor):
                    optimizer2.zero_grad()
                    if (args.hybrid==False and args.epsylon_greedy==False):
                        optimizer3.zero_grad()
                if (args.quantum_critic):
                    optimizer4.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                #if (args.quantum_actor):
                #    nn.utils.clip_grad_norm_([actor_layer_params], args.max_grad_norm)
                #if (args.quantum_critic):
                #    nn.utils.clip_grad_norm_([critic_layer_params], args.max_grad_norm)
                optimizer1.step()
                if (args.quantum_actor):
                    optimizer2.step()
                    if (args.hybrid==False and args.epsylon_greedy==False):
                        optimizer3.step()
                if (args.quantum_critic):
                    optimizer4.step()

    	    # Early stopping on to high target kl
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # debug variables (info)
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y           # is the value function a good indicator for the returns


        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer1.param_groups[0]["lr"], global_step)
        if (args.quantum_actor):
            writer.add_scalar("charts/qlearning_rate", optimizer2.param_groups[0]["lr"], global_step)
        elif (args.quantum_critic):
            writer.add_scalar("charts/qlearning_rate", optimizer4.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/loss", loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        #if (args.quantum_actor and args.hybrid==False):
        #    #print("actor_layer_params", actor_layer_params)
        #    print("observable_rescaleing_params:", output_scaleing_params)
        
        if (args.hybrid==False and args.epsylon_greedy==False and args.quantum_actor):
            writer.add_scalar("greedyness/output_scaleing", output_scaleing_params.mean()**2, global_step)
            #writer.add_scalar("greedyness/output_scaleing_var", np.var(output_scaleing_params), global_step)
        if (args.epsylon_greedy):
            writer.add_scalar("greedyness/epsylon", epsylon, global_step)

        store_envs.store_envs(actions, dones, args.num_steps, args.num_envs)
        if(update % args.save_intervall == 0):
            if (args.quantum_actor==False or args.quantum_critic==False):
                agent.save_classical_network_params()
            if (args.quantum_actor):
                save_params.save_actor_circuit_params(args.chkpt_dir, actor_layer_params)
            if (args.quantum_actor and args.hybrid==False):    
                save_params.save_output_scaleing_params(args.chkpt_dir, output_scaleing_params)
            if (args.quantum_critic):
                save_params.save_critic_circuit_params(args.chkpt_dir, critic_layer_params)
            save_params.save_state(args.chkpt_dir, global_step, next_obs, next_done)
            #save_envs(args.chkpt_dir, envs)
            store_envs.save_envs(args.chkpt_dir, args.num_envs)
            #store_envs.load_envs(args.chkpt_dir, args.num_envs)
            #print("store_envs.get_storage(0)", store_envs.get_storage(0))
            #print("store_envs.get_storage(1)", store_envs.get_storage(1))
            #print("store_envs.get_storage(2)", store_envs.get_storage(2))
            #print("store_envs.get_storage(3)", store_envs.get_storage(3))



    envs.close()
    writer.close()
