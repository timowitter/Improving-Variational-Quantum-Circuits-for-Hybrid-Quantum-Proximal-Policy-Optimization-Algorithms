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

# plot gradients


def plot_gradient_avg(results_dir, plot_dir, gym_id, exp_names, seeds, max_steps):
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
    gr_res_df_list = [
        pd.read_csv(os.path.join(loc, "gradient_results.csv")) for loc in results_dirs
    ]

    # concat dataframes
    gr_res = pd.concat(gr_res_df_list, ignore_index=True)
    gr_res.drop(gr_res[gr_res["global_step"] > max_steps].index, inplace=True)

    # check if plot_dir exists
    pathExists = os.path.exists(plot_dir)
    if not pathExists:
        os.makedirs(plot_dir)

    plot_actor_grads_abs_mean_by_seed(gr_res, plot_dir, exp_names, seeds)
    plot_actor_grads_abs_mean_by_exp_name(gr_res, plot_dir, exp_names)
    plot_actor_gradients_var_by_seed(gr_res, plot_dir)
    plot_actor_gradients_var_by_exp_name(gr_res, plot_dir)


def lower(row):
    row["lower"] = row["actor_grads_abs_mean"] - row["actor_gradients_std"]
    return row


def upper(row):
    row["upper"] = row["actor_grads_abs_mean"] + row["actor_gradients_std"]
    return row


def lower_exp_name(row):
    row["lower"] = row["actor_grads_abs_mean_exp_name"] - row["actor_gradients_std_exp_name"]
    return row


def upper_exp_name(row):
    row["upper"] = row["actor_grads_abs_mean_exp_name"] + row["actor_gradients_std_exp_name"]
    return row


def plot_actor_grads_abs_mean_by_seed(data, dir, exp_names, seeds, smoothing_fac=0.6):
    dataset = data.drop(
        columns=[
            "actor_gradients_var",
            "critic_grads_abs_mean",
            "critic_gradients_var",
            "critic_gradients_std",
            "circuit",
        ]
    )

    dataset_tmp = dataset.groupby(["exp_name", "seed", "global_step"], sort=False).apply(lower)
    dataset_tmp.reset_index(drop=True, inplace=True)
    dataset_plt = dataset_tmp.groupby(["exp_name", "seed", "global_step"], sort=False).apply(upper)

    # smoothing
    # dataset_plt["actor_grads_abs_mean"] = (
    #    dataset_plt["actor_grads_abs_mean"].ewm(alpha=smoothing_fac).mean()
    # )
    # dataset_plt["lower"] = dataset_plt["lower"].ewm(alpha=smoothing_fac).mean()
    # dataset_plt["upper"] = dataset_plt["upper"].ewm(alpha=smoothing_fac).mean()

    ax = sns.lineplot(
        data=dataset_plt,
        x="global_step",
        y="actor_grads_abs_mean",
        style="exp_name",
        hue="seed",
        errorbar=None,
    )
    for exp_name in exp_names:
        for seed in seeds:
            dataset_seed = dataset_plt[
                (dataset_plt.exp_name == exp_name) & (dataset_plt.seed == seed)
            ]
            ax.fill_between(
                dataset_seed.global_step, dataset_seed.lower, dataset_seed.upper, alpha=0.2
            )
    plot_dir = os.path.join(dir, "actor_grads_abs_mean_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_actor_grads_abs_mean_by_exp_name(data, dir, exp_names, smoothing_fac=0.6):
    dataset = data.drop(
        columns=[
            "actor_gradients_var",
            "critic_grads_abs_mean",
            "critic_gradients_var",
            "critic_gradients_std",
            "circuit",
        ]
    )
    dataset["actor_grads_abs_mean_exp_name"] = dataset.groupby(
        ["exp_name", "global_step"], sort=False
    )["actor_grads_abs_mean"].transform("mean")

    dataset["actor_gradients_std_exp_name"] = dataset.groupby(
        ["exp_name", "global_step"], sort=False
    )["actor_gradients_std"].transform("mean")

    dataset_tmp = (
        dataset.groupby(["exp_name", "global_step"], sort=False)
        .apply(lower_exp_name)
        .drop(columns=["actor_grads_abs_mean", "actor_gradients_std"])
    )
    dataset_tmp.reset_index(drop=True, inplace=True)
    dataset_plt = (
        dataset_tmp.groupby(["exp_name", "global_step"], sort=False)
        .apply(upper_exp_name)
        .drop(columns=["actor_gradients_std_exp_name"])
    )

    # smoothing
    # dataset_plt["actor_grads_abs_mean_exp_name"] = (
    #    dataset_plt["actor_grads_abs_mean_exp_name"].ewm(alpha=smoothing_fac).mean()
    # )
    # dataset_plt["lower"] = dataset_plt["lower"].ewm(alpha=smoothing_fac).mean()
    # dataset_plt["upper"] = dataset_plt["upper"].ewm(alpha=smoothing_fac).mean()

    ax = sns.lineplot(
        data=dataset_plt,
        x="global_step",
        y="actor_grads_abs_mean_exp_name",
        hue="exp_name",
        errorbar=None,
    )
    for exp_name in exp_names:
        dataset_exp_name = dataset_plt[(dataset_plt.exp_name == exp_name)]
        ax.fill_between(
            dataset_exp_name.global_step, dataset_exp_name.lower, dataset_exp_name.upper, alpha=0.2
        )
    plot_dir = os.path.join(dir, "actor_grads_abs_mean_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_actor_gradients_var_by_seed(data, dir, smoothing_fac=0.6):
    dataset = data.drop(
        columns=[
            "actor_gradients_std",
            "critic_grads_abs_mean",
            "critic_gradients_var",
            "critic_gradients_std",
            "circuit",
        ]
    )
    # smoothing
    # dataset["actor_gradients_var"] = dataset["actor_gradients_var"].ewm(alpha=smoothing_fac).mean()

    sns.lineplot(
        data=dataset,
        x="global_step",
        y="actor_gradients_var",
        style="exp_name",
        hue="seed",
        errorbar=None,
    )

    plot_dir = os.path.join(dir, "actor_gradients_var_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_actor_gradients_var_by_exp_name(data, dir, smoothing_fac=0.6):
    dataset = data.drop(
        columns=[
            "actor_gradients_std",
            "critic_grads_abs_mean",
            "critic_gradients_var",
            "critic_gradients_std",
            "circuit",
        ]
    )
    dataset["actor_gradients_var_exp_name"] = dataset.groupby(
        ["exp_name", "global_step"], sort=False
    )["actor_gradients_var"].transform("mean")

    # smoothing
    # dataset["actor_gradients_var_exp_name"] = (
    #    dataset["actor_gradients_var_exp_name"].ewm(alpha=smoothing_fac).mean()
    # )

    sns.lineplot(
        data=dataset,
        x="global_step",
        y="actor_gradients_var_exp_name",
        hue="exp_name",
        errorbar="sd",
    )

    plot_dir = os.path.join(dir, "actor_gradients_var_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


##################################################################################
##################################################################################
##################################################################################

# plot insider info


def plot_abs_cart_velocity_by_seed(data, dir):
    sns.relplot(
        data=data,
        kind="scatter",
        x="global_step",
        y="abs_cart_velocity",
        col="exp_name",
        hue="seed",
    )
    plot_dir = os.path.join(dir, "abs_cart_velocity_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_abs_pole_velocity_by_seed(data, dir):
    sns.relplot(
        data=data,
        kind="scatter",
        x="global_step",
        y="abs_pole_velocity",
        col="exp_name",
        hue="seed",
    )
    plot_dir = os.path.join(dir, "abs_pole_velocity_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_abs_cart_velocity_mean(data, dir):
    sns.relplot(
        data=data,
        kind="line",
        x="global_step",
        y="abs_cart_velocity_mean_emw",
        col="gym_id",
        hue="exp_name",
        # errorbar="sd",
        # err_style='band',
    )
    plot_dir = os.path.join(dir, "abs_cart_velocity_mean.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_abs_pole_velocity_mean(data, dir):
    sns.relplot(
        data=data,
        kind="line",
        x="global_step",
        y="abs_pole_velocity_mean_emw",
        col="gym_id",
        hue="exp_name",
        # errorbar="sd",
        # err_style='band',
    )
    plot_dir = os.path.join(dir, "abs_pole_velocity_mean.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_insider_by_seed(data, dir):
    data["abs_cart_velocity_mean"] = data.groupby(["exp_name", "global_step"], sort=False)[
        "abs_cart_velocity"
    ].transform("mean")

    data["abs_cart_velocity_mean_emw"] = data["abs_cart_velocity_mean"].ewm(alpha=0.6).mean()

    data["abs_pole_velocity_mean"] = data.groupby(["exp_name", "global_step"], sort=False)[
        "abs_pole_velocity"
    ].transform("mean")

    data["abs_pole_velocity_mean_emw"] = data["abs_pole_velocity_mean"].ewm(alpha=0.6).mean()

    avg_abs_cart_velocity = data.groupby(["exp_name"], sort=False)["abs_cart_velocity"].transform(
        "mean"
    )

    print(avg_abs_cart_velocity)

    avg_abs_pole_velocity = data.groupby(["exp_name"], sort=False)["abs_pole_velocity"].transform(
        "mean"
    )

    print(avg_abs_pole_velocity)

    #
    plot_abs_cart_velocity_by_seed(data, dir)
    plot_abs_pole_velocity_by_seed(data, dir)
    plot_abs_cart_velocity_mean(data, dir)
    plot_abs_pole_velocity_mean(data, dir)


def plot_insider_info(info_dir, plot_dir, gym_id, exp_names, seeds, max_steps):
    # get all result directories
    info_dirs = []
    dir_seeds = []
    dir_exp_names = []
    for exp_name in exp_names:
        for seed in seeds:
            info_dirs.append(info_dir + "/" + gym_id + "/" + exp_name + "/" + str(seed))
            dir_seeds.append(seed)
            dir_exp_names.append(exp_name)

    # load dataframes
    df_list = [pd.read_csv(os.path.join(loc, "insider_info.csv")) for loc in info_dirs]

    # concat dataframes
    data = pd.concat(df_list, ignore_index=True)
    data.drop(data[data["global_step"] > max_steps].index, inplace=True)

    # check if plot_dir exists
    pathExists = os.path.exists(plot_dir)
    if not pathExists:
        os.makedirs(plot_dir)

    plot_insider_by_seed(data, plot_dir)


##################################################################################
##################################################################################
##################################################################################
