import argparse
import json
import os
from distutils.util import strtobool


def save_args(args):
    args_chkpt_file = os.path.join(args.chkpt_dir, "commandline_args.txt")
    # if os.path.exists(args_chkpt_file): # delete old file with same name if it exists and make a new run
    #        os.remove(args_chkpt_file)
    with open(args_chkpt_file, "w") as f:
        json.dump(args.__dict__, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment",
    )
    parser.add_argument(
        "--gym-id",
        type=str,
        default="CartPole-v1",
        help="id of the OpenAi Gym Environment /default: CartPole-v1",
    )
    parser.add_argument(
        "--exp-scheduling-qlearning-rate",
        type=float,
        default=5e-3,
        help="quantum optimizer learning rate at start of exp-scheduling",
    )
    parser.add_argument(
        "--lin-scheduling-qlearning-rate",
        type=float,
        default=1e-5,
        help="quantum optimizer learning rate after lin-scheduling",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2.5e-4, help="optimizer learning rate"
    )  # 2.5e-4
    parser.add_argument(
        "--qlearning-rate", type=float, default=1e-3, help="quantum optimizer learning rate"
    )
    parser.add_argument(
        "--output-scaleing-learning-rate",
        type=float,
        default=5e-4,
        help="output scaleing learning rate",
    )
    parser.add_argument(
        "--output-scaleing-start", type=float, default=1.0, help="output scaleing start value"
    )
    parser.add_argument("--seed", type=int, default=1, help="seed for random events")
    parser.add_argument(
        "--total-timesteps", type=int, default=50000, help="length of the Training"
    )  # 25000
    parser.add_argument(
        "--exp-scheduling-timesteps",
        type=int,
        default=50000,
        help="length of the exp-scheduling Phase",
    )
    parser.add_argument(
        "--lin-scheduling-timesteps",
        type=int,
        default=500000,
        help="length of the lin-scheduling Phase (not executed during exp-scheduling timesteps if both are enabled)",
    )
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="cdnn.deterministic use /default:True",
    )
    # parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True, help='determines if gpu shall be used /default:False')
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="wether to record videos to `videos` folder",
    )  # not working

    # Algorith specific
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="number of environments run in parrallel in the SyncVectorEnv",
    )  # 1 Vector environment
    parser.add_argument(
        "--num-steps",
        type=int,
        default=128,
        help="the number of steps in each environments per policy rollout phase of the multy vector env",
    )
    parser.add_argument(
        "--exp-qlr-scheduling",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="exponential leaning rate scheduling for quantum circuits, executed before lin-qlr-scheduling if both are enabled",
    )
    parser.add_argument(
        "--lin-qlr-scheduling",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="linear leaning rate scheduling for quantum circuits, executed after exp-qlr-scheduling if both are enabled",
    )
    parser.add_argument(
        "--gae",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="GAE for Advantage computation",
    )  # 5 general advantage estimation
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor gamma")  # 0.98
    parser.add_argument(
        "--gae-lambda", type=float, default=0.95, help="lsmbda for general advantage estimation"
    )
    parser.add_argument(
        "--num-minibatches", type=int, default=4, help="number of minibatches"
    )  # 6 minibatches
    parser.add_argument(
        "--update-epochs", type=int, default=4, help="the k epochs to update the policy"
    )
    parser.add_argument(
        "--norm-adv",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="toggle Advantage normalisation",
    )  # 7 Advatage Normalisation
    parser.add_argument(
        "--clip-coef", type=float, default=0.2, help="clipping coefficient"
    )  # 8 clipped Objektive
    parser.add_argument(
        "--clip-vloss",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="toggle the use of clipped value loss",
    )  # 9 Value loss clipping
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="coefficient for the entropy (loss), 0 = disabled",
    )  # 10 entropy loss
    parser.add_argument(
        "--vf-coef", type=float, default=0.5, help="coefficient for the value function"
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=0.5, help="maximum norm for the gradient clipping"
    )  # 11 global gradient clipping
    parser.add_argument(
        "--target-kl", type=float, default=None, help="the target KL divergence threshhold (0.015)"
    )  # target KL early stopping

    parser.add_argument(
        "--actor-hidden-layer-nodes",
        type=int,
        default=64,
        help="number of nodes of the 2 hidden layers of the actor",
    )
    parser.add_argument(
        "--critic-hidden-layer-nodes",
        type=int,
        default=64,
        help="number of nodes of the 2 hidden layers of the critic",
    )
    parser.add_argument(
        "--quantum-actor",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if True quantum actor will be used",
    )
    parser.add_argument(
        "--quantum-critic",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if True quantum critic will be used",
    )
    parser.add_argument("--n-qubits", type=int, default=4, help="number of qubits of the circuit")
    parser.add_argument(
        "--n-var-layers",
        type=int,
        default=2,
        help="gives the number of variational layers in case of the simple circuit",
    )
    parser.add_argument(
        "--n-enc-layers",
        type=int,
        default=1,
        help="gives the the number of encodeing layers in case of data reuploading (then the number of variational layers equals the number of encodeing layers + 1)",
    )
    parser.add_argument(
        "--circuit",
        type=str,
        default="simple",
        help="the circuit that is to be used, can be one of:  simple / simple_reuploading / simple_reuploading_with_input_scaleing / Hgog / Hgog_reuploading / Hgog_reuploading_with_output_scaleing / Jerbi-no-reuploading-no-input-scaleing / Jerbi-reuploading-no-input-scaleing / Jerbi-reuploading",
    )
    parser.add_argument(
        "--output-scaleing",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="toggle output scaleing",
    )
    parser.add_argument(
        "--hybrid",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="toggle hybrid output postprocessing",
    )
    parser.add_argument(
        "--load-chkpt",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=False,
        help="continue learning from checkpoint",
    )
    parser.add_argument(
        "--save-location",
        type=str,
        default="qppo-slurm",
        help="save-location/gym-id/exp-name/seed/ make the save location of the experiment (chkpt-dir)",
    )
    parser.add_argument(
        "--save-intervall",
        type=int,
        default=20,
        help="gives the the save-intervall in number of update epochs",
    )
    parser.add_argument(
        "--scheduled-output-scaleing",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="toggle scheduled output scaleing or trainable output scaleing",
    )
    parser.add_argument(
        "--sced-out-scale-fac",
        type=float,
        default=2.0,
        help="value output scaleing is increased with every/over 100000 steps",
    )
    parser.add_argument(
        "--random-baseline",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="toggle completely random action selection",
    )

    parser.add_argument(
        "--log-circuit-output",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if no output scaleing or hybrid active toggle if circuit output is used as logprob for softmax instead of prob",
    )

    parser.add_argument(
        "--clip-circuit-grad-norm",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="toggle clip circuit gradient norm",
    )

    parser.add_argument(
        "--record-grads",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="toggle clip circuit gradient norm",
    )
    parser.add_argument(
        "--param-init",
        type=str,
        default="random",
        help="the parameter initialisation that is to be used, can be one of:  random / random_clipped / gauss_distribution",
    )

    args = parser.parse_args()
    args.batch_size = int(args.num_steps * args.num_envs)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.chkpt_dir = f"{args.save_location}/checkpoints/{args.gym_id}/{args.exp_name}/{args.seed}"
    args.results_dir = f"{args.save_location}/results/{args.gym_id}/{args.exp_name}/{args.seed}"
    args.plot_dir = f"{args.save_location}/plots/{args.gym_id}/{args.exp_name}/{args.seed}"

    if (
        args.load_chkpt
    ):  # load args exept total_timesteps and continue learning if new total_timesteps > global_timesteps
        args_chkpt_file = os.path.join(args.chkpt_dir, "commandline_args.txt")
        t = args.total_timesteps
        with open(args_chkpt_file, "r") as f:
            args.__dict__ = json.load(f)
        args.load_chkpt = True
        args.total_timesteps = t

    return args
