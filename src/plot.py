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
# plt.rcParams["text.usetex"] = True
# rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
# rc("text", usetex=True)
# sns.color_palette("colorblind")

# palette = itertools.cycle(sns.color_palette("colorblind"))  # type: ignore


def avg_episode_results_of_update(episode_results_df, gym_id, exp_name, seed, stepsize, max_steps):
    # make avg over update for plotting
    avg_rewards = []
    avg_lengths = []
    avg_global_step = []
    avg_gym_id = []
    avg_exp_name = []
    # avg_circuit = []
    avg_seed = []

    for i in range(stepsize, max_steps, stepsize):
        df_tmp = episode_results_df[
            (episode_results_df.global_step <= i)
            & (episode_results_df.global_step > (i - stepsize))
        ]
        avg_rewards.append(df_tmp["episode_reward"].mean())
        avg_lengths.append(df_tmp["episode_length"].mean())
        avg_global_step.append(i)
        avg_gym_id.append(gym_id)
        avg_exp_name.append(exp_name)
        # avg_circuit.append(circuit)
        avg_seed.append(seed)

    avg_results = {
        "avg_rewards": avg_rewards,
        "avg_lengths": avg_lengths,
        "global_step": avg_global_step,
        "gym_id": gym_id,
        "exp_name": avg_exp_name,
        #'circuit': circuit,
        "seed": avg_seed,
    }

    return pd.DataFrame(data=avg_results)


def avg_episode_results_of_update_by_exp_name(df, gym_id, exp_names, stepsize, max_steps):
    avg_rewards = []
    avg_lengths = []
    avg_global_step = []
    avg_gym_id = []
    avg_exp_name = []

    for exp_name in exp_names:
        df_tmp1 = df[(df.exp_name == exp_name)]
        for i in range(stepsize, max_steps, stepsize):
            df_tmp2 = df_tmp1[(df_tmp1.global_step == i)]
            avg_rewards.append(df_tmp2["avg_rewards"].mean())
            avg_lengths.append(df_tmp2["avg_lengths"].mean())
            avg_global_step.append(i)
            avg_gym_id.append(gym_id)
            avg_exp_name.append(exp_name)

    avg_results = {
        "avg_rewards": avg_rewards,
        "avg_lengths": avg_lengths,
        "global_step": avg_global_step,
        "gym_id": avg_gym_id,
        "exp_name": avg_exp_name,
    }

    return pd.DataFrame(data=avg_results)


def avg_update_results(df, gym_id, exp_names, stepsize, max_steps):
    avg_learning_rate = []
    avg_qlearning_rate = []
    avg_value_loss = []
    avg_policy_loss = []
    avg_entropy = []
    avg_loss = []
    avg_SPS = []
    avg_output_scaleing = []

    avg_global_step = []
    avg_gym_id = []
    avg_exp_name = []

    for exp_name in exp_names:
        df_tmp1 = df[(df.exp_name == exp_name)]
        for i in range(stepsize, max_steps, stepsize):
            df_tmp = df_tmp1[(df_tmp1.global_step <= i) & (df_tmp1.global_step > (i - stepsize))]
            avg_learning_rate.append(df_tmp["learning_rate"].mean())
            avg_qlearning_rate.append(df_tmp["qlearning_rate"].mean())
            avg_value_loss.append(df_tmp["value_loss"].mean())
            avg_policy_loss.append(df_tmp["policy_loss"].mean())
            avg_entropy.append(df_tmp["entropy"].mean())
            avg_loss.append(df_tmp["loss"].mean())
            avg_SPS.append(df_tmp["SPS"].mean())
            avg_output_scaleing.append(df_tmp["output_scaleing"].mean())

            avg_global_step.append(i)
            avg_gym_id.append(gym_id)
            avg_exp_name.append(exp_name)

    avg_results = {
        "learning_rate": avg_learning_rate,
        "qlearning_rate": avg_qlearning_rate,
        "value_loss": avg_value_loss,
        "policy_loss": avg_policy_loss,
        "entropy": avg_entropy,
        "loss": avg_loss,
        "SPS": avg_SPS,
        "output_scaleing": avg_output_scaleing,
        "global_step": avg_global_step,
        "gym_id": avg_gym_id,
        "exp_name": avg_exp_name,
    }

    return pd.DataFrame(data=avg_results)


def plot_training_results(
    results_dir, plot_dir, gym_id, exp_name, circuit, seed, stepsize, max_steps
):
    pathExists = os.path.exists(plot_dir)
    if not pathExists:
        os.makedirs(plot_dir)

    episode_results_dir = os.path.join(results_dir, "episode_results.csv")
    episode_results = pd.read_csv(episode_results_dir)

    update_results_dir = os.path.join(results_dir, "update_results.csv")
    update_results = pd.read_csv(update_results_dir)

    avg_episode_results = avg_episode_results_of_update(
        episode_results, gym_id, exp_name, seed, stepsize, max_steps
    )

    # plotting over update
    plot_avg_episode_reward_by_seed(avg_episode_results, plot_dir)
    plot_avg_episode_length_by_seed(avg_episode_results, plot_dir)

    plot_learning_rate_by_seed(update_results, plot_dir)
    plot_qlearning_rate_by_seed(update_results, plot_dir)
    plot_value_loss_by_seed(update_results, plot_dir)
    plot_policy_loss_by_seed(update_results, plot_dir)
    plot_entropy_by_seed(update_results, plot_dir)
    plot_loss_by_seed(update_results, plot_dir)
    plot_old_approx_kl_by_seed(update_results, plot_dir)
    plot_approx_kl_by_seed(update_results, plot_dir)
    plot_clipfrac_by_seed(update_results, plot_dir)
    plot_explained_variance_by_seed(update_results, plot_dir)
    plot_SPS_by_seed(update_results, plot_dir)
    plot_output_scaleing_by_seed(update_results, plot_dir)


def plot_avg_episode_reward_by_seed(episode_results, dir):
    sns.relplot(
        data=episode_results,
        kind="line",
        x="global_step",
        y="avg_rewards",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "episode_reward_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_avg_episode_reward_by_exp_name(episode_results, dir):
    sns.relplot(
        data=episode_results,
        kind="line",
        x="global_step",
        y="avg_rewards",
        col="gym_id",
        hue="exp_name",
        # style="exp_name",
    )
    plot_dir = os.path.join(dir, "episode_reward_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_avg_episode_length_by_seed(episode_results, dir):
    sns.relplot(
        data=episode_results,
        kind="line",
        x="global_step",
        y="avg_lengths",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "episode_lenght_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_avg_episode_length_by_exp_name(episode_results, dir):
    sns.relplot(
        data=episode_results,
        kind="line",
        x="global_step",
        y="avg_lengths",
        col="gym_id",
        hue="exp_name",
        # style="exp_name",
    )
    plot_dir = os.path.join(dir, "episode_lenght_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_learning_rate_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="learning_rate",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "learning_rate_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_learning_rate_by_exp_name(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="learning_rate",
        col="gym_id",
        hue="exp_name",
        # style="exp_name",
    )
    plot_dir = os.path.join(dir, "learning_rate_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_qlearning_rate_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="qlearning_rate",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "qlearning_rate_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_qlearning_rate_by_exp_name(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="qlearning_rate",
        col="gym_id",
        hue="exp_name",
        # style="exp_name",
    )
    plot_dir = os.path.join(dir, "qlearning_rate_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_value_loss_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="value_loss",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "value_loss_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_value_loss_by_exp_name(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="value_loss",
        col="gym_id",
        hue="exp_name",
        # style="exp_name",
    )
    plot_dir = os.path.join(dir, "value_loss_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_policy_loss_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="policy_loss",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "policy_loss_by.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_policy_loss_by_exp_name(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="policy_loss",
        col="gym_id",
        hue="exp_name",
        # style="exp_name",
    )
    plot_dir = os.path.join(dir, "policy_loss_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_entropy_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="entropy",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "entropy_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_entropy_by_exp_name(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="entropy",
        col="gym_id",
        hue="exp_name",
        # style="exp_name",
    )
    plot_dir = os.path.join(dir, "entropy_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_loss_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="loss",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "loss_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_loss_by_exp_name(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="loss",
        col="gym_id",
        hue="exp_name",
        # style="exp_name",
    )
    plot_dir = os.path.join(dir, "loss_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_old_approx_kl_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="old_approx_kl",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "old_approx_kl_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_approx_kl_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="approx_kl",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "approx_kl_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_clipfrac_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="clipfrac",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "clipfrac_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_explained_variance_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="explained_variance",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "explained_variance_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_SPS_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="SPS",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "SPS_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_SPS_by_exp_name(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="SPS",
        col="gym_id",
        hue="exp_name",
        # style="exp_name",
    )
    plot_dir = os.path.join(dir, "SPS_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_output_scaleing_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="output_scaleing",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "output_scaleing_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_output_scaleing_by_exp_name(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="output_scaleing",
        col="gym_id",
        hue="exp_name",
        # style="exp_name",
    )
    plot_dir = os.path.join(dir, "output_scaleing_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, stepsize, max_steps):
    # get all result directories
    results_dirs = []
    dir_seeds = []
    dir_exp_names = []
    for exp_name in exp_names:
        for seed in seeds:
            # tmp = f"{results_dir}/{gym_id}/{exp_name}/{seed}"
            results_dirs.append(results_dir + "/" + gym_id + "/" + exp_name + "/" + str(seed))
            dir_seeds.append(seed)
            dir_exp_names.append(exp_name)

    # load dataframes
    ep_res_df_list = [
        pd.read_csv(os.path.join(loc, "episode_results.csv")) for loc in results_dirs
    ]
    up_res_df_list = [pd.read_csv(os.path.join(loc, "update_results.csv")) for loc in results_dirs]
    # make average of update for episode specific data
    ep_res_avg_list = [
        avg_episode_results_of_update(df, gym_id, exp_name, seed, stepsize, max_steps)
        for df, seed, exp_name in zip(ep_res_df_list, dir_seeds, dir_exp_names)
    ]
    # concat dataframes
    ep_res = pd.concat(ep_res_avg_list, ignore_index=True)
    up_res = pd.concat(up_res_df_list, ignore_index=True)
    # make average over the different seeds of one experiment
    ep_res_avg = avg_episode_results_of_update_by_exp_name(
        ep_res, gym_id, exp_names, stepsize, max_steps
    )
    up_res_avg = avg_update_results(up_res, gym_id, exp_names, stepsize, max_steps)

    # check if plot_dir exists
    pathExists = os.path.exists(plot_dir)
    if not pathExists:
        os.makedirs(plot_dir)

    # plotting
    plot_avg_episode_reward_by_seed(ep_res, plot_dir)
    plot_avg_episode_reward_by_exp_name(ep_res_avg, plot_dir)
    plot_avg_episode_length_by_seed(ep_res, plot_dir)
    plot_avg_episode_length_by_exp_name(ep_res_avg, plot_dir)

    plot_learning_rate_by_exp_name(up_res_avg, plot_dir)
    plot_qlearning_rate_by_exp_name(up_res_avg, plot_dir)
    plot_value_loss_by_exp_name(up_res_avg, plot_dir)
    plot_policy_loss_by_exp_name(up_res_avg, plot_dir)
    plot_entropy_by_seed(up_res, plot_dir)
    plot_entropy_by_exp_name(up_res_avg, plot_dir)
    plot_loss_by_exp_name(up_res_avg, plot_dir)
    plot_old_approx_kl_by_seed(up_res, plot_dir)
    plot_approx_kl_by_seed(up_res, plot_dir)
    plot_clipfrac_by_seed(up_res, plot_dir)
    plot_explained_variance_by_seed(up_res, plot_dir)
    plot_SPS_by_seed(up_res, plot_dir)
    plot_SPS_by_exp_name(up_res_avg, plot_dir)
    plot_output_scaleing_by_seed(up_res, plot_dir)
    plot_output_scaleing_by_exp_name(up_res_avg, plot_dir)
