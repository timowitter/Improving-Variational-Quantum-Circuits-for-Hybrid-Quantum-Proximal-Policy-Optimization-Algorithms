import itertools
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from matplotlib import rc

mpl.use("Agg")
sns.set_theme()

# Apply the default theme
#plt.rcParams["text.usetex"] = True
#rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
#rc("text", usetex=True)
sns.color_palette("colorblind")

#palette = itertools.cycle(sns.color_palette("colorblind"))  # type: ignore

# plot final results


def plot_test_avg_final(
    results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, namelabels, batchsize=512
):
    # get all result directories
    results_dirs = []
    dir_seeds = []
    dir_exp_names = []
    for exp_name, namelabel in zip(exp_names, namelabels):
        for seed in seeds:
            results_dirs.append(results_dir + "/" + gym_id + "/" + exp_name + "/" + str(seed))
            dir_seeds.append(seed)
            dir_exp_names.append(namelabel)

    # load dataframes
    ep_res_df_list = [
        pd.read_csv(os.path.join(loc, "episode_results.csv")) for loc in results_dirs
    ]
    for df in ep_res_df_list:
        df.drop(df[df["global_step"] > max_steps].index, inplace=True)

    up_res_df_list = [pd.read_csv(os.path.join(loc, "update_results.csv")) for loc in results_dirs]
    # make average of update for episode specific data
    for df in up_res_df_list:
        df.drop(df[df["global_step"] > max_steps].index, inplace=True)

    for df in up_res_df_list:
        for i in range(np.size(exp_names)):
            df.loc[df['exp_name'] == exp_names[i], 'exp_name'] = namelabels[i]

    ep_res_by_seed_list = [
        avg_over_update_of_episodic_results(df, gym_id, exp_name, seed, batchsize, max_steps)
        for df, seed, exp_name in zip(ep_res_df_list, dir_seeds, dir_exp_names)
    ]

    ep_res_by_seed_ema_list = [
        ema_for_plotting_by_seed_of_episodic_results(df, alpha) for df in ep_res_by_seed_list
    ]

    ep_res_by_seed = pd.concat(ep_res_by_seed_ema_list, ignore_index=True)

    up_res_by_seed_ema_list = [
        ema_for_plotting_by_seed_of_update_results(df, alpha) for df in up_res_df_list
    ]

    up_res_by_seed = pd.concat(up_res_by_seed_ema_list, ignore_index=True)

    #ep_res_by_exp_name_list = [
    #    avg_over_seeds_of_avg_over_update_of_episodic_results(
    #        df, gym_id, exp_name, 512, max_steps, alpha
    #    )
    #    for df, exp_name in zip(ep_res_by_seed_list, dir_exp_names)
    #]
    #ep_res_by_exp_name = pd.concat(ep_res_by_exp_name_list, ignore_index=True)

    #up_res_by_exp_name_list = [
    #    avg_over_seeds_of_update_results(df, gym_id, exp_name, batchsize, max_steps, alpha)
    #    for df, exp_name in zip(up_res_df_list, dir_exp_names)
    #]

    #up_res_by_exp_name = pd.concat(up_res_by_exp_name_list, ignore_index=True)

    # check if plot_dir exists
    pathExists = os.path.exists(plot_dir)
    if not pathExists:
        os.makedirs(plot_dir)

    # plotting

    plot_avg_episode_reward_by_seed(ep_res_by_seed, plot_dir)
    plot_avg_episode_reward_by_exp_name(ep_res_by_seed, plot_dir)
    plot_avg_episode_length_by_seed(ep_res_by_seed, plot_dir)
    plot_avg_episode_length_by_exp_name(ep_res_by_seed, plot_dir)

    plot_learning_rate_by_exp_name(up_res_by_seed, plot_dir)
    plot_qlearning_rate_by_exp_name(up_res_by_seed, plot_dir)
    #plot_value_loss_by_exp_name(up_res_by_seed, plot_dir)
    #plot_policy_loss_by_exp_name(up_res_by_seed, plot_dir)
    #plot_entropy_by_seed(up_res_by_seed, plot_dir)
    plot_entropy_by_exp_name(up_res_by_seed, plot_dir)
    #plot_loss_by_exp_name(up_res_by_seed, plot_dir)
    #plot_old_approx_kl_by_seed(up_res_by_seed, plot_dir)
    #plot_approx_kl_by_seed(up_res_by_seed, plot_dir)
    #plot_clipfrac_by_seed(up_res_by_seed, plot_dir)
    #plot_explained_variance_by_seed(up_res_by_seed, plot_dir)
    plot_SPS_by_seed(up_res_by_seed, plot_dir)
    plot_output_scaleing_by_seed(up_res_by_seed, plot_dir)
    plot_output_scaleing_by_exp_name(up_res_by_seed, plot_dir)

##################################################################################
##################################################################################
##################################################################################


def avg_over_update_of_episodic_results(
    episode_results_df, gym_id, exp_name, seed, batchsize, max_steps
):
    # make avg over update for makeing the mean over different seeds
    avg_reward_per_updates = []
    avg_episode_length_per_updates = []
    avg_global_step = []
    avg_gym_id = []
    avg_exp_name = []
    # avg_circuit = []
    avg_seed = []

    for i in range(batchsize, max_steps, batchsize):
        df_tmp = episode_results_df[
            (episode_results_df.global_step <= i)
            & (episode_results_df.global_step > (i - batchsize))
        ]
        avg_reward_per_updates.append(df_tmp["episode_reward"].mean())
        avg_episode_length_per_updates.append(df_tmp["episode_length"].mean())
        avg_global_step.append(i)
        avg_gym_id.append(gym_id)
        avg_exp_name.append(exp_name)
        # avg_circuit.append(circuit)
        avg_seed.append(seed)

    avg_results = {
        "avg_reward_per_update": avg_reward_per_updates,
        "avg_episode_length_per_update": avg_episode_length_per_updates,
        "global_step": avg_global_step,
        "gym_id": gym_id,
        "exp_name": avg_exp_name,
        #'circuit': circuit,
        "seed": avg_seed,
    }

    return pd.DataFrame(data=avg_results)


def ema_for_plotting_by_seed_of_episodic_results(df, alpha):
    df["reward"] = df["avg_reward_per_update"].ewm(alpha=alpha).mean()
    df["episode_length"] = df["avg_episode_length_per_update"].ewm(alpha=alpha).mean()
    return df


def avg_over_seeds_of_avg_over_update_of_episodic_results(
    df, gym_id, exp_name, batchsize, max_steps, alpha
):  
    
    avg_rewards = []
    avg_lengths = []
    avg_global_step = []
    avg_gym_id = []
    avg_exp_name = []

    df_tmp1 = df[(df.exp_name == exp_name)]
    for i in range(batchsize, max_steps, batchsize):
        df_tmp2 = df_tmp1[(df_tmp1.global_step == i)]
        avg_rewards.append(df_tmp2["avg_reward_per_update"].mean())
        avg_lengths.append(df_tmp2["avg_episode_length_per_update"].mean())
        avg_global_step.append(i)
        avg_gym_id.append(gym_id)
        avg_exp_name.append(exp_name)

    avg_results = {
        "avg_reward_no_ema": avg_rewards,
        "avg_length_no_ema": avg_lengths,
        "global_step": avg_global_step,
        "gym_id": avg_gym_id,
        "exp_name": avg_exp_name,
    }
    avg_ep_res_avg = pd.DataFrame(data=avg_results)
    avg_ep_res_avg["reward"] = avg_ep_res_avg["avg_reward_no_ema"].ewm(alpha=alpha).mean()
    avg_ep_res_avg["episode_length"] = avg_ep_res_avg["avg_length_no_ema"].ewm(alpha=alpha).mean()

    return avg_ep_res_avg


def ema_for_plotting_by_seed_of_update_results(df, alpha):
    df["value_loss_ema"] = df["value_loss"].ewm(alpha=alpha).mean()
    df["policy_loss_ema"] = df["policy_loss"].ewm(alpha=alpha).mean()
    df["entropy_ema"] = df["entropy"].ewm(alpha=alpha).mean()
    df["loss_ema"] = df["loss"].ewm(alpha=alpha).mean()
    return df


def avg_over_seeds_of_update_results(df, gym_id, exp_name, batchsize, max_steps, alpha):
    avg_learning_rate = []
    avg_qlearning_rate = []
    avg_value_loss = []
    avg_policy_loss = []
    avg_entropy = []
    avg_loss = []
    avg_output_scaleing = []

    avg_global_step = []
    avg_gym_id = []
    avg_exp_name = []

    df_tmp1 = df[(df.exp_name == exp_name)]
    for i in range(batchsize, max_steps, batchsize):
        df_tmp = df_tmp1[(df_tmp1.global_step <= i) & (df_tmp1.global_step > (i - batchsize))]
        avg_learning_rate.append(df_tmp["learning_rate"].mean())
        avg_qlearning_rate.append(df_tmp["qlearning_rate"].mean())
        avg_value_loss.append(df_tmp["value_loss"].mean())
        avg_policy_loss.append(df_tmp["policy_loss"].mean())
        avg_entropy.append(df_tmp["entropy"].mean())
        avg_loss.append(df_tmp["loss"].mean())
        avg_output_scaleing.append(df_tmp["output_scaleing"].mean())

        avg_global_step.append(i)
        avg_gym_id.append(gym_id)
        avg_exp_name.append(exp_name)

    avg_results = {
        "avg_learning_rate": avg_learning_rate,
        "avg_qlearning_rate": avg_qlearning_rate,
        "avg_value_loss": avg_value_loss,
        "avg_policy_loss": avg_policy_loss,
        "avg_entropy": avg_entropy,
        "avg_loss": avg_loss,
        "avg_output_scaleing": avg_output_scaleing,
        "global_step": avg_global_step,
        "gym_id": avg_gym_id,
        "exp_name": avg_exp_name,
    }

    avg_up_res_avg = pd.DataFrame(data=avg_results)
    avg_up_res_avg["value_loss_ema"] = avg_up_res_avg["avg_value_loss"].ewm(alpha=alpha).mean()
    avg_up_res_avg["policy_loss_ema"] = avg_up_res_avg["avg_policy_loss"].ewm(alpha=alpha).mean()
    avg_up_res_avg["entropy_ema"] = avg_up_res_avg["avg_entropy"].ewm(alpha=alpha).mean()
    avg_up_res_avg["loss_ema"] = avg_up_res_avg["avg_loss"].ewm(alpha=alpha).mean()

    return avg_up_res_avg


def plot_avg_episode_reward_by_seed(episode_results, plot_dir):
    g = sns.relplot(
        data=episode_results,
        kind="line",
        x="global_step",
        y="reward",
        col="exp_name",

        hue="seed",
    )
    g._legend.set_title("Seed")
    g.set(xlabel ="Zeitschritt", ylabel = "Reward")
    #g.set(xlabel ="Globaler Schritt", ylabel = "Reward", title ='some title')
    #sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    #sns.move_legend(g, "upper left")
    plot_dir = os.path.join(plot_dir, "episode_reward_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_avg_episode_reward_by_exp_name(episode_results, plot_dir):
    g=sns.relplot(
        data=episode_results,
        kind="line",
        x="global_step",
        y="reward",

        hue="exp_name",
    )
    g._legend.set_title("Ansatz")
    g.set(xlabel ="Zeitschritt", ylabel = "Reward")
    #for t, l in zip(g._legend.texts, labels):
        #t.set_text(l)
    #sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    #sns.move_legend(g, "upper left")
    plot_dir = os.path.join(plot_dir, "episode_reward_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_avg_episode_length_by_seed(episode_results, plot_dir):
    g = sns.relplot(
        data=episode_results,
        kind="line",
        x="global_step",
        y="episode_length",
        col="exp_name",
        hue="seed",
    )
    g._legend.set_title("Seed")
    g.set(xlabel ="Zeitschritt", ylabel = "Episodenlänge")
    plot_dir = os.path.join(plot_dir, "episode_lenght_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_avg_episode_length_by_exp_name(episode_results, plot_dir):
    g=sns.relplot(
        data=episode_results,
        kind="line",
        x="global_step",
        y="episode_length",
        
        hue="exp_name",
    )
    g._legend.set_title("Ansatz")
    g.set(xlabel ="Zeitschritt", ylabel = "Episodenlänge")
    #for t, l in zip(g._legend.texts, labels):
        #t.set_text(l)
    plot_dir = os.path.join(plot_dir, "episode_lenght_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_learning_rate_by_seed(update_results, plot_dir):
    g = sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="learning_rate",
        col="exp_name",
        hue="seed",
    )
    g._legend.set_title("Seed")
    g.set(xlabel ="Zeitschritt", ylabel = "Actor NN Lernrate")
    plot_dir = os.path.join(plot_dir, "learning_rate_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_learning_rate_by_exp_name(update_results, plot_dir):
    g = sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="learning_rate",
        
        hue="exp_name",
    )
    g._legend.set_title("Ansatz")
    g.set(xlabel ="Zeitschritt", ylabel = "Actor NN Lernrate")
    #for t, l in zip(g._legend.texts, labels):
        #t.set_text(l)
    plot_dir = os.path.join(plot_dir, "learning_rate_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_qlearning_rate_by_seed(update_results, plot_dir):
    g = sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="qlearning_rate",
        col="exp_name",
        hue="seed",
    )
    g._legend.set_title("Seed")
    g.set(xlabel ="Zeitschritt", ylabel = "Actor VQC Lernrate")
    plot_dir = os.path.join(plot_dir, "qlearning_rate_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_qlearning_rate_by_exp_name(update_results, plot_dir):
    g = sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="qlearning_rate",
        
        hue="exp_name",
    )
    g._legend.set_title("Ansatz")
    g.set(xlabel ="Zeitschritt", ylabel = "Actor VQC Lernrate")
    #for t, l in zip(g._legend.texts, labels):
        #t.set_text(l)
    plot_dir = os.path.join(plot_dir, "qlearning_rate_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_value_loss_by_seed(update_results, plot_dir):
    g = sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="value_loss_ema",
        col="exp_name",
        hue="seed",
    )
    g._legend.set_title("Seed")
    g.set(xlabel ="Zeitschritt", ylabel = "Value Loss")
    plot_dir = os.path.join(plot_dir, "value_loss_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_value_loss_by_exp_name(update_results, plot_dir):
    g = sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="value_loss_ema",
        
        hue="exp_name",
    )
    g._legend.set_title("Ansatz")
    g.set(xlabel ="Zeitschritt", ylabel = "Value Loss")
    #for t, l in zip(g._legend.texts, labels):
        #t.set_text(l)
    plot_dir = os.path.join(plot_dir, "value_loss_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_policy_loss_by_seed(update_results, plot_dir):
    g = sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="policy_loss_ema",
        col="exp_name",
        hue="seed",
    )
    g._legend.set_title("Seed")
    g.set(xlabel ="Zeitschritt", ylabel = "Policy Loss")
    plot_dir = os.path.join(plot_dir, "policy_loss_by.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_policy_loss_by_exp_name(update_results, plot_dir):
    g = sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="policy_loss_ema",
        
        hue="exp_name",
    )
    g._legend.set_title("Ansatz")
    g.set(xlabel ="Zeitschritt", ylabel = "Policy Loss")
    #for t, l in zip(g._legend.texts, labels):
        #t.set_text(l)
    plot_dir = os.path.join(plot_dir, "policy_loss_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_entropy_by_seed(update_results, plot_dir):
    g = sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="entropy_ema",
        col="exp_name",
        hue="seed",
    )
    g._legend.set_title("Seed")
    g.set(xlabel ="Zeitschritt", ylabel = "Entropie")
    plot_dir = os.path.join(plot_dir, "entropy_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_entropy_by_exp_name(update_results, plot_dir):
    g = sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="entropy_ema",
        
        hue="exp_name",
    )
    g._legend.set_title("Ansatz")
    g.set(xlabel ="Zeitschritt", ylabel = "Entropie")
    #for t, l in zip(g._legend.texts, labels):
        #t.set_text(l)
    plot_dir = os.path.join(plot_dir, "entropy_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_loss_by_seed(update_results, plot_dir):
    g = sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="loss_ema",
        col="exp_name",
        hue="seed",
    )
    g._legend.set_title("Seed")
    g.set(xlabel ="Zeitschritt", ylabel = "Loss")
    plot_dir = os.path.join(plot_dir, "loss_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_loss_by_exp_name(update_results, plot_dir):
    g = sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="loss_ema",
        
        hue="exp_name",
    )
    g._legend.set_title("Ansatz")
    g.set(xlabel ="Zeitschritt", ylabel = "Loss")
    #for t, l in zip(g._legend.texts, labels):
        #t.set_text(l)
    plot_dir = os.path.join(plot_dir, "loss_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_old_approx_kl_by_seed(update_results, plot_dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="old_approx_kl",
        col="exp_name",
        hue="seed",
    )

    plot_dir = os.path.join(plot_dir, "old_approx_kl_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_approx_kl_by_seed(update_results, plot_dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="approx_kl",
        col="exp_name",
        hue="seed",
    )

    plot_dir = os.path.join(plot_dir, "approx_kl_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_clipfrac_by_seed(update_results, plot_dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="clipfrac",
        col="exp_name",
        hue="seed",
    )

    plot_dir = os.path.join(plot_dir, "clipfrac_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_explained_variance_by_seed(update_results, plot_dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="explained_variance",
        col="exp_name",
        hue="seed",
    )

    plot_dir = os.path.join(plot_dir, "explained_variance_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_SPS_by_seed(update_results, plot_dir):
    g = sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="SPS",
        col="exp_name",
        hue="seed",
    )
    g._legend.set_title("Seed")
    g.set(xlabel ="Zeitschritt", ylabel = "Zeitschritte pro Sekunde")
    plot_dir = os.path.join(plot_dir, "SPS_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_output_scaleing_by_seed(update_results, plot_dir):
    g = sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="output_scaleing",
        col="exp_name",
        hue="seed",
    )
    g._legend.set_title("Seed")
    g.set(xlabel ="Zeitschritt", ylabel = "Output Scaleing")
    plot_dir = os.path.join(plot_dir, "output_scaleing_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_output_scaleing_by_exp_name(update_results, plot_dir):
    g = sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="output_scaleing",
        
        hue="exp_name",
    )
    g._legend.set_title("Ansatz")
    g.set(xlabel ="Zeitschritt", ylabel = "Output Scaleing")
    #for t, l in zip(g._legend.texts, labels):
        #t.set_text(l)
    plot_dir = os.path.join(plot_dir, "output_scaleing_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()
