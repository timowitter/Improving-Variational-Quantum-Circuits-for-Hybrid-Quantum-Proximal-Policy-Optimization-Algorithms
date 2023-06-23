import gym
import os
import json
import copy
import torch
import pickle
import numpy as np
from envs_storage import Store_envs

#Chen's shortest path frozen lake (from https://github.com/ycchen1989/Var-QuantumCircuits-DeepRL/tree/master/Code)
from gym.envs.registration import register
register(
    id='Deterministic-ShortestPath-4x4-FrozenLake-v0', # name given to this new environment
    entry_point='ShortestPathFrozenLake:ShortestPathFrozenLake', # env entry point
    kwargs={'desc': ["SFFF", "FFFH", "FHFH", "HFFG"], 'map_name': '4x4', 'is_slippery': False} # argument passed to the env
)
# register(
#     id='Deterministic-4x4-FrozenLake-v0', # name given to this new environment
#     entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv', # env entry point
#     kwargs={'map_name': '4x4', 'is_slippery': False} # argument passed to the env
# )

"""
#environment setup:
def make_env(gym_id, seed, env_num, capture_video, run_name):
    def thunk():
        if gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0":
            env = gym.make('Deterministic-ShortestPath-4x4-FrozenLake-v0')
            print("ShortestPathFrozenLake")
            env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
        elif (gym_id == "FrozenLake-v1"):
            env = gym.make(gym_id, desc=["SFFF", "FFFH", "FHFH", "HFFG"], map_name="4x4", is_slippery=False)
            print("is_slippery=False")
            env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
        else:
            env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if env_num==0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", record_video_trigger=lambda t: t % 1000 == 0)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk
"""

#environment setup:
def make_env(gym_id, seed, env_num, capture_video, run_name, num_envs, chkpt_dir, load_chkpt, store_envs):
    def thunk():
        if gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0":
            env = gym.make('Deterministic-ShortestPath-4x4-FrozenLake-v0')
            print("ShortestPathFrozenLake")
            env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
        elif (gym_id == "FrozenLake-v1"):
            env = gym.make(gym_id, desc=["SFFF", "FFFH", "FHFH", "HFFG"], map_name="4x4", is_slippery=False)
            print("is_slippery=False")
            env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
        else:
            env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if env_num==0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", record_video_trigger=lambda t: t % 1000 == 0)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        if (load_chkpt and store_envs.restore_envs[env_num]):
            env.reset()
            store_envs.load_envs(chkpt_dir, num_envs)
            checkpoint_aktions = store_envs.get_storage(env_num)
            if (checkpoint_aktions.size == 1):
                env.step(int(checkpoint_aktions))
            else:
                for j in range(checkpoint_aktions.size):
                    env.step(checkpoint_aktions[j])
            store_envs.restore_envs[env_num] = False
        return env
    return thunk


#todo: save environment states of the multi vector environment
"""
def save_envs(chkpt_dir, envs):
    save_point_envs = copy.deepcopy(envs)           #copy.deepcopy(envs)
    args_chkpt_file = os.path.join(chkpt_dir, 'envs.txt')
    with open(args_chkpt_file, 'wb') as f:
        pickle.dump(save_point_envs, f)     #json.dump

#if (load_chkpt):  
def load_envs(chkpt_dir):        # load envs from file          
    args_chkpt_file = os.path.join(chkpt_dir, 'envs.txt') 
    with open(args_chkpt_file, 'rb') as f:
        envs = pickle.load(f)
    return envs
"""



"""
def save_envs(chkpt_dir, envs):
    save_point_envs = copy.deepcopy(envs)
    envs_chkpt_file = os.path.join(chkpt_dir, 'envs.txt')
    if (os.path.exists(envs_chkpt_file)):
        os.remove(envs_chkpt_file)
    file = open(envs_chkpt_file, 'w')
    file.write(save_point_envs)
    file.close()

#if (load_chkpt):  
def load_envs(chkpt_dir):        # load envs from file          
    envs_chkpt_file = os.path.join(chkpt_dir, 'envs.txt') 
    file = open(envs_chkpt_file, 'r')
    envs = file.read()
    file.close()
    return envs
"""







"""
def save_envs(chkpt_dir, envs):
    save_point_envs = copy.deepcopy(envs)
    args_chkpt_file = os.path.join(chkpt_dir, 'envs')
    torch.save(save_point_envs, args_chkpt_file)

#if (load_chkpt):  
def load_envs(chkpt_dir):        # load envs from file          
    args_chkpt_file = os.path.join(chkpt_dir, 'envs') 
    envs = torch.load(args_chkpt_file)
    return envs
"""
