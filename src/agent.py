import numpy as np
import os
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from transform_funks import trans_obs
from args import parse_args
args = parse_args()
#######################################################################################################################################################
##                                                  agent:                                                                                           ##
##                                                                                                                                                   ##

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):                                                                                                        #2 Layer Initialisation
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        if (args.quantum_critic==False and args.gym_id == "FrozenLake-v0" or args.gym_id == "FrozenLake-v1" or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"):
            self.critic = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.n).prod(), args.critic_hidden_layer_nodes)), 
                nn.Tanh(),
                layer_init(nn.Linear(args.critic_hidden_layer_nodes, args.critic_hidden_layer_nodes)), 
                nn.Tanh(),
                layer_init(nn.Linear(args.critic_hidden_layer_nodes, 1), std=1.),
            )
        elif(args.quantum_critic==False):
            self.critic = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), args.critic_hidden_layer_nodes)), 
                nn.Tanh(),
                layer_init(nn.Linear(args.critic_hidden_layer_nodes, args.critic_hidden_layer_nodes)), 
                nn.Tanh(),
                layer_init(nn.Linear(args.critic_hidden_layer_nodes, 1), std=1.),
            )
        
        if (args.quantum_actor==False and args.gym_id == "FrozenLake-v0" or args.gym_id == "FrozenLake-v1" or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"):
            self.actor = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.n).prod(), args.actor_hidden_layer_nodes)), 
                nn.Tanh(),
                layer_init(nn.Linear(args.actor_hidden_layer_nodes, args.actor_hidden_layer_nodes)), 
                nn.Tanh(),
                layer_init(nn.Linear(args.actor_hidden_layer_nodes, envs.single_action_space.n), std=0.01),
            )
        elif(args.quantum_actor==False):
            self.actor = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), args.actor_hidden_layer_nodes)), 
                nn.Tanh(),
                layer_init(nn.Linear(args.actor_hidden_layer_nodes, args.actor_hidden_layer_nodes)), 
                nn.Tanh(),
                layer_init(nn.Linear(args.actor_hidden_layer_nodes, envs.single_action_space.n), std=0.01),
            )
        
        if(args.hybrid):
            self.hybrid_actor = nn.Sequential(
                layer_init(nn.Linear(args.n_qubits, envs.single_action_space.n))
            )
            self.hybrid_critic = nn.Sequential(
                layer_init(nn.Linear(args.n_qubits, 1))
            )

        self.checkpoint_file = os.path.join(args.chkpt_dir, 'classical_network_params')

    def save_classical_network_params(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_classical_network_params(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

    def get_value(self, observation, obs_dim, critic_circuit, critic_layer_params):
        if args.quantum_critic:                 #quantum critic
            crit_tmp = critic_circuit(critic_layer_params, observation)

            if (args.hybrid):                   #hybrid output rescaleing
                crit = self.hybrid_critic(crit_tmp.float())
            else:                               #manual output rescaleing
                if (args.gym_id == "FrozenLake-v0" or args.gym_id == "FrozenLake-v1"):
                    crit = (crit_tmp + 1) / 2     #rescaleing from [-1,1] to [0,1]
                elif(args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"):
                    crit = crit_tmp               #intervall [-1,1]
                elif(args.gym_id == "CartPole-v0"):
                    crit = (crit_tmp + 1) * 100   #rescaleing from [-1,1] to [0,200]
                elif(args.gym_id == "CartPole-v1"):
                    crit = (crit_tmp + 1) * 250   #rescaleing from [-1,1] to [0,500]
                else:
                    print("output rescaleing ERROR")
                    crit = crit_tmp
        else:                                   #classical critic
            crit = self.critic(trans_obs(observation, args.gym_id, obs_dim))

        return crit
    
    def get_action_and_value(self, observation, actor_circuit, actor_layer_params, critic_circuit, critic_layer_params, output_scaleing_params, obs_dim, acts_dim, action=None):  #, envs.single_observation_space.n, envs.single_action_space.n
        
        if args.quantum_actor:
            logits_i = actor_circuit(actor_layer_params, observation, acts_dim)
        else:
            logits_i = self.actor(trans_obs(observation, args.gym_id, obs_dim)) 

        #output scaleing
        if (args.quantum_actor and args.hybrid==False and args.output_scaleing):
            # trainable output parameters for trainable greedyness
            logits=torch.zeros(acts_dim) 
            for i in range(acts_dim):
                logits[i]=logits_i[i]*(output_scaleing_params[i]*output_scaleing_params.mean())
        elif (args.quantum_actor and args.hybrid):
            logits = self.hybrid_actor(logits_i.float())
        else:
            logits = logits_i

        probs = Categorical(logits=logits)      #softMaxOutPut = np.exp(logits) / np.exp(logits).sum()
        if action is None:
            """
            if args.epsylon_greedy:
                if (np.random.random(1) <= epsylon):
                    action=torch.tensor(np.random.randint(0, acts_dim)) 
                else: 
                    action=torch.argmax(logits)
            else:"""
            action = probs.sample()
            action = action.view(1)

        return action , probs.log_prob(action), probs.entropy(), self.get_value(observation, obs_dim, critic_circuit, critic_layer_params)
##                                                                                                                                                   ##
##                                                  agent:                                                                                           ##
#######################################################################################################################################################