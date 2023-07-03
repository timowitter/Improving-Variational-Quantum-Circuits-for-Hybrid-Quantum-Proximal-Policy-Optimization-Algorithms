import os
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc

import seaborn as sns
import seaborn.objects as so

mpl.use('Agg')
sns.set_theme()

# Apply the default theme
#plt.rcParams["text.usetex"] = True
#rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
#rc("text", usetex=True)
#sns.color_palette("colorblind")

#palette = itertools.cycle(sns.color_palette("colorblind"))  # type: ignore



def plot_training_results(results_dir, plot_dir, exp_name, seed, stepsize, max_steps):
    pathExists = os.path.exists(plot_dir)
    if not pathExists:
        os.makedirs(plot_dir)

    episode_results_dir = os.path.join(results_dir, 'episode_results.csv')
    episode_results = pd.read_csv(episode_results_dir)

    update_results_dir = os.path.join(results_dir, 'update_results.csv')
    update_results = pd.read_csv(update_results_dir)

    #make avg over update for plotting
    avg_rewards=[]
    avg_lengths=[]
    avg_global_step=[]
    avg_exp_name=[]
    avg_seed=[]
    
    for i in range(stepsize, max_steps, stepsize):
        df_tmp = episode_results[(episode_results.global_step <= i) & (episode_results.global_step > (i-stepsize))]
        avg_rewards.append(df_tmp['episode_reward'].mean())
        avg_lengths.append(df_tmp['episode_length'].mean())
        avg_global_step.append(i)
        avg_exp_name.append(exp_name)
        avg_seed.append(seed)
    
    avg_results = {
                            'avg_rewards': avg_rewards,
                            'avg_lengths': avg_lengths,
                            'global_step': avg_global_step,
                            'exp_name': avg_exp_name,
                            'seed': avg_seed
        }
    
    avg_episode_results = pd.DataFrame(data=avg_results)


    #plotting over update
    plot_avg_episode_reward(avg_episode_results, plot_dir)
    plot_avg_episode_length(avg_episode_results, plot_dir)

    plot_learning_rate(update_results, plot_dir)
    plot_qlearning_rate(update_results, plot_dir)
    plot_value_loss(update_results, plot_dir)
    plot_policy_loss(update_results, plot_dir)
    plot_entropy(update_results, plot_dir)
    plot_loss(update_results, plot_dir)
    plot_old_approx_kl(update_results, plot_dir)
    plot_approx_kl(update_results, plot_dir)
    plot_clipfrac(update_results, plot_dir)
    plot_explained_variance(update_results, plot_dir)
    plot_SPS(update_results, plot_dir)
    plot_output_scaleing(update_results, plot_dir)




def plot_avg_episode_reward(episode_results, dir):
    sns.relplot(
    data=episode_results, kind="line",
    x="global_step", y="avg_rewards", col="exp_name",
    hue="seed", style="seed",
    )
    plot_dir = os.path.join(dir, 'episode_reward.png')
    plt.savefig(plot_dir)


def plot_avg_episode_length(episode_results, dir):
    sns.relplot(
    data=episode_results, kind="line",
    x="global_step", y="avg_lengths", col="exp_name",
    hue="seed", style="seed",
    )
    plot_dir = os.path.join(dir, 'episode_lenght.png')
    plt.savefig(plot_dir)


def plot_learning_rate(update_results, dir):
    sns.relplot(
    data=update_results, kind="line",
    x="global_step", y="learning_rate", col="exp_name",
    hue="seed", style="seed",
    )
    plot_dir = os.path.join(dir, 'learning_rate.png')
    plt.savefig(plot_dir)

def plot_qlearning_rate(update_results, dir):
    sns.relplot(
    data=update_results, kind="line",
    x="global_step", y="qlearning_rate", col="exp_name",
    hue="seed", style="seed",
    )
    plot_dir = os.path.join(dir, 'qlearning_rate.png')
    plt.savefig(plot_dir)

def plot_value_loss(update_results, dir):
    sns.relplot(
    data=update_results, kind="line",
    x="global_step", y="value_loss", col="exp_name",
    hue="seed", style="seed",
    )
    plot_dir = os.path.join(dir, 'value_loss.png')
    plt.savefig(plot_dir)

def plot_policy_loss(update_results, dir):
    sns.relplot(
    data=update_results, kind="line",
    x="global_step", y="policy_loss", col="exp_name",
    hue="seed", style="seed",
    )
    plot_dir = os.path.join(dir, 'policy_loss.png')
    plt.savefig(plot_dir)

def plot_entropy(update_results, dir):
    sns.relplot(
    data=update_results, kind="line",
    x="global_step", y="entropy", col="exp_name",
    hue="seed", style="seed",
    )
    plot_dir = os.path.join(dir, 'entropy.png')
    plt.savefig(plot_dir)

def plot_loss(update_results, dir):
    sns.relplot(
    data=update_results, kind="line",
    x="global_step", y="loss", col="exp_name",
    hue="seed", style="seed",
    )
    plot_dir = os.path.join(dir, 'loss.png')
    plt.savefig(plot_dir)

def plot_old_approx_kl(update_results, dir):
    sns.relplot(
    data=update_results, kind="line",
    x="global_step", y="old_approx_kl", col="exp_name",
    hue="seed", style="seed",
    )
    plot_dir = os.path.join(dir, 'old_approx_kl.png')
    plt.savefig(plot_dir)

def plot_approx_kl(update_results, dir):
    sns.relplot(
    data=update_results, kind="line",
    x="global_step", y="approx_kl", col="exp_name",
    hue="seed", style="seed",
    )
    plot_dir = os.path.join(dir, 'approx_kl.png')
    plt.savefig(plot_dir)

def plot_clipfrac(update_results, dir):
    sns.relplot(
    data=update_results, kind="line",
    x="global_step", y="clipfrac", col="exp_name",
    hue="seed", style="seed",
    )
    plot_dir = os.path.join(dir, 'clipfrac.png')
    plt.savefig(plot_dir)

def plot_explained_variance(update_results, dir):
    sns.relplot(
    data=update_results, kind="line",
    x="global_step", y="explained_variance", col="exp_name",
    hue="seed", style="seed",
    )
    plot_dir = os.path.join(dir, 'explained_variance.png')
    plt.savefig(plot_dir)

def plot_SPS(update_results, dir):
    sns.relplot(
    data=update_results, kind="line",
    x="global_step", y="SPS", col="exp_name",
    hue="seed", style="seed",
    )
    plot_dir = os.path.join(dir, 'SPS.png')
    plt.savefig(plot_dir)

def plot_output_scaleing(update_results, dir):
    sns.relplot(
    data=update_results, kind="line",
    x="global_step", y="output_scaleing", col="exp_name",
    hue="seed", style="seed",
    )
    plot_dir = os.path.join(dir, 'output_scaleing.png')
    plt.savefig(plot_dir)



def extract_from_df_by_exp_name(df, exp_name):
    df_extract=df[df['exp_name'] == exp_name]
    return df_extract

def get_avg_of_episode(df, exp_names, seeds, stepsize, max_steps):
    avg_rewards=[]
    avg_lengths=[]
    avg_global_step=[]
    avg_exp_name=[]
    
    for exp_name in exp_names:
        df_tmp1 = df[(df.exp_name == exp_name)]
        for i in range(stepsize, max_steps, stepsize):
            df_tmp2 = df_tmp1[(df_tmp1.global_step <= i) & (df_tmp1.global_step > (i-stepsize))]
            seed_avg_reward=[]
            seed_avg_length=[]
            for seed in seeds:
                df_tmp = df_tmp2[(df_tmp2.seed == seed)]
                print(df_tmp)
                seed_avg_reward.append(df_tmp['episode_reward'].mean())
                seed_avg_length.append(df_tmp['episode_length'].mean())
            avg_rewards.append(np.mean(seed_avg_reward))
            avg_lengths.append(np.mean(seed_avg_length))
            avg_global_step.append(i)
            avg_exp_name.append(exp_name)
    
    avg_results = {
                            'avg_rewards': avg_rewards,
                            'avg_lengths': avg_lengths,
                            'global_step': avg_global_step,
                            'exp_name': avg_exp_name,
        }
    
    return pd.DataFrame(data=avg_results)

    

def plot_multiple(results_dir, dir):
    #episode_results_dir = os.path.join(results_dir, 'episode_results.csv')
    #episode_results = pd.read_csv(episode_results_dir)
    #episode_results['reward_smoothed'] = episode_results['episode_reward'].rolling(1000).sum()/1000
    print("1")
    
    ep_res_dir1 = os.path.join('d:\Studium\Skripts\Skrips Neu\qppo-slurm/results/Deterministic-ShortestPath-4x4-FrozenLake-v0/ppo_plot_actor_test/1', 'episode_results.csv')
    ep_res1 = pd.read_csv(ep_res_dir1)
    ep_res_dir2 = os.path.join('d:\Studium\Skripts\Skrips Neu\qppo-slurm/results/Deterministic-ShortestPath-4x4-FrozenLake-v0/ppo_plot_actor_test/10', 'episode_results.csv')
    ep_res2 = pd.read_csv(ep_res_dir2)
    ep_res_dir3 = os.path.join('d:\Studium\Skripts\Skrips Neu\qppo-slurm/results/Deterministic-ShortestPath-4x4-FrozenLake-v0/ppo_plot_actor_test/100', 'episode_results.csv')
    ep_res3 = pd.read_csv(ep_res_dir3)
    ep_res_dir4 = os.path.join('d:\Studium\Skripts\Skrips Neu\qppo-slurm/results/Deterministic-ShortestPath-4x4-FrozenLake-v0/ppo_plot_critic_test/1', 'episode_results.csv')
    ep_res4 = pd.read_csv(ep_res_dir4)
    ep_res_dir5 = os.path.join('d:\Studium\Skripts\Skrips Neu\qppo-slurm/results/Deterministic-ShortestPath-4x4-FrozenLake-v0/ppo_plot_critic_test/10', 'episode_results.csv')
    ep_res5 = pd.read_csv(ep_res_dir5)
    ep_res_dir6 = os.path.join('d:\Studium\Skripts\Skrips Neu\qppo-slurm/results/Deterministic-ShortestPath-4x4-FrozenLake-v0/ppo_plot_critic_test/100', 'episode_results.csv')
    ep_res6 = pd.read_csv(ep_res_dir6)
    
    

    df=pd.concat([ep_res1, ep_res2, ep_res3, ep_res4, ep_res5, ep_res6], ignore_index=True)
    seeds=[1, 10, 100]
    names = ["ppo_plot_actor_test", "ppo_plot_critic_test"]
    avg_results=get_avg_of_episode(df, names, seeds, 1000, 100000)

    print("2")

    ep_res1['reward_smoothed'] = ep_res1['episode_reward'].rolling(100).mean()  #.sum()/100
    ep_res2['reward_smoothed'] = ep_res2['episode_reward'].rolling(100).mean()  #.sum()/100
    ep_res3['reward_smoothed'] = ep_res3['episode_reward'].rolling(100).mean()  #.sum()/100
    ep_res4['reward_smoothed'] = ep_res4['episode_reward'].rolling(100).mean()  #.sum()/100
    ep_res5['reward_smoothed'] = ep_res5['episode_reward'].rolling(100).mean()  #.sum()/100
    ep_res6['reward_smoothed'] = ep_res6['episode_reward'].rolling(100).mean()  #.sum()/100
    
    episode_results = pd.concat([ep_res1, ep_res2, ep_res3, ep_res4, ep_res5, ep_res6], ignore_index=True)

    
    
    
    
    print("3")
    
    sns.relplot(
    data=episode_results,
    x="global_step", y="episode_reward", col="exp_name",
    hue="seed", style="seed",
    )
    
    plot_dir = os.path.join(dir, 'episode_reward.png')
    plt.savefig(plot_dir)

    print("4")

    sns.relplot(
    data=episode_results, kind="line",
    x="global_step", y="reward_smoothed", col="exp_name",
    hue="seed", style="seed",
    )

    plot_dir = os.path.join(dir, 'reward_smoothed.png')
    plt.savefig(plot_dir)

    print("5")

    sns.relplot(
    data=avg_results, kind="line",
    x="global_step", y="avg_rewards", col="exp_name",
    hue="exp_name", style="exp_name"
    )
    print("6")
    plot_dir = os.path.join(dir, 'reward_smoothed_avg.png')
    plt.savefig(plot_dir)

    print("7")
    







#episode_results_avg1 = pd.concat([ep_res1, ep_res2, ep_res3], ignore_index=True)
    #episode_results_avg2 = pd.concat([ep_res4, ep_res5, ep_res6], ignore_index=True)
    #episode_results_avg1['reward_smoothed_avg'] = episode_results_avg1.groupby('global_step')['episode_reward'].rolling(10000, min_periods=1).mean().reset_index(0,drop=True)
    #episode_results_avg2['reward_smoothed_avg'] = episode_results_avg2.groupby('global_step')['episode_reward'].rolling(10000, min_periods=1).mean().reset_index(0,drop=True)
    #episode_results_avg = pd.concat([episode_results_avg1, episode_results_avg2], ignore_index=True)

#episode_results2 = episode_results.drop(episode_results[episode_results.episode_reward == np.nan].index)
#print("episode_results_avg['reward_smoothed_avg']", episode_results_avg['reward_smoothed_avg'])
    #episode_results_avg = episode_results.drop('seed', axis=1).groupby("global_step")
    #episode_results_avg['reward_smoothed_avg'] = episode_results_avg['episode_reward'].rolling(100).mean()#.sum()/100

    #for i in ["ppo_plot_actor_test", "ppo_plot_critic_test"]:
    #    episode_results['reward_smoothed_avg'] = extract_from_df_by_exp_name(episode_results, i)['episode_reward'].rolling(100).sum()/100

    #episode_results_avg1 = extract_from_df_by_exp_name(episode_results, "ppo_plot_actor_test")
    #episode_results_avg1['reward_smoothed2'] = episode_results_avg1['episode_reward'].rolling(100).sum()/100                                     #[['global_step', 'reward']].copy()
    #episode_results_avg2 = extract_from_df_by_exp_name(episode_results, "ppo_plot_critic_test")
    #episode_results_avg2['reward_smoothed2'] = episode_results_avg2 ['episode_reward'].rolling(100).sum()/100                                    #[['global_step', 'reward']].copy()












"""

# Load an example dataset
tips = sns.load_dataset("tips")

# Create a visualization
sns.relplot(
    data=tips,
    x="total_bill", y="tip", col="time",
    hue="smoker", style="smoker", size="size",
)


dots = sns.load_dataset("dots")
sns.relplot(
    data=dots, kind="line",
    x="time", y="firing_rate", col="align",
    hue="choice", size="coherence", style="choice",
    facet_kws=dict(sharex=False),
)
"""



