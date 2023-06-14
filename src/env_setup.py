import gym

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


#environment setup:
def make_env(gym_id, seed, idx, capture_video, run_name):
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
            if idx==0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", record_video_trigger=lambda t: t % 1000 == 0)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk