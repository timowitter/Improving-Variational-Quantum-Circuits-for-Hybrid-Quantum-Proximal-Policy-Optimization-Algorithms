from args import parse_args

args = parse_args()

# number of parameters for a neural network layer: number_of_inputs * number_of_outputs + number_of_outputs
#                                                 (             weights              ) + (     biases    )

# number of parameters for a quantum circuit:      qubits * layers * dimensions   +   (output scaleing   or   hybrid neural network)


# calculate number of trainable actor parameters
###############################################################
def calc_num_actor_params():
    actor_par_count = 0
    if (
        args.gym_id == "FrozenLake-v0"
        or args.gym_id == "FrozenLake-v1"
        or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"
    ):
        num_obs = 16
        num_acts = 4
        debug = 1
    elif args.gym_id == "CartPole-v0" or args.gym_id == "CartPole-v1":
        num_obs = 4
        num_acts = 2
        debug = 1
    else:
        num_obs = 0
        num_acts = 0
        debug = 0
        print("number of actor parameters calculation error")

    if args.quantum_actor and (
        args.circuit == "simple"
        or args.circuit == "Hgog"
        or args.circuit == "simple_reuploading"
        or args.circuit == "Hgog_reuploading"
    ):
        actor_par_count = 3 * args.n_var_layers * args.n_qubits
    elif args.quantum_actor and (
        args.circuit == "simple_reuploading_with_input_scaleing"
        or args.circuit == "Hgog_reuploading_with_input_scaleing"
    ):
        actor_par_count = (
            3 * args.n_var_layers * args.n_qubits + 1 * args.n_enc_layers * args.n_qubits
        )
    elif args.quantum_actor and (
        args.circuit == "Jerbi-no-reuploading-no-input-scaleing"
        or args.circuit == "Jerbi-reuploading-no-input-scaleing"
    ):
        actor_par_count = 2 * args.n_var_layers * args.n_qubits
    elif args.quantum_actor and args.circuit == "Jerbi-reuploading":
        actor_par_count = 2 * (2 * args.n_enc_layers + 1) * args.n_qubits
    elif not args.quantum_actor:
        actor_par_count = (
            num_obs * args.actor_hidden_layer_nodes
            + args.actor_hidden_layer_nodes**2
            + args.actor_hidden_layer_nodes * num_acts
            + 2 * args.actor_hidden_layer_nodes
            + num_acts
        )

    if args.quantum_actor and args.hybrid:
        actor_par_count = actor_par_count + args.n_qubits * num_acts + num_acts
    elif (
        args.quantum_actor
        and not args.hybrid
        and args.output_scaleing
        and not args.scheduled_output_scaleing
    ):
        if not args.shared_output_scaleing_param:
            actor_par_count = actor_par_count + num_acts
        else:
            actor_par_count = actor_par_count + 1
    actor_par_count = actor_par_count * debug
    return actor_par_count


###############################################################


# calculate number of trainable critic parameters
###############################################################
def calc_num_critic_params():
    critic_par_count = 0
    if (
        args.gym_id == "FrozenLake-v0"
        or args.gym_id == "FrozenLake-v1"
        or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"
    ):
        num_obs = 16
        debug = 1
    elif args.gym_id == "CartPole-v0" or args.gym_id == "CartPole-v1":
        num_obs = 4
        debug = 1
    else:
        num_obs = 0
        debug = 0
        print("number of critic parameters calculation error")

    if args.quantum_critic and (
        args.circuit == "simple"
        or args.circuit == "Hgog"
        or args.circuit == "simple_reuploading"
        or args.circuit == "Hgog_reuploading"
    ):
        critic_par_count = 3 * args.n_var_layers * args.n_qubits
    elif args.quantum_actor and (
        args.circuit == "simple_reuploading_with_input_scaleing"
        or args.circuit == "Hgog_reuploading_with_input_scaleing"
    ):
        critic_par_count = (
            3 * args.n_var_layers * args.n_qubits + 1 * args.n_enc_layers * args.n_qubits
        )
    elif args.quantum_critic and (
        args.circuit == "Jerbi-no-reuploading-no-input-scaleing"
        or args.circuit == "Jerbi-reuploading-no-input-scaleing"
    ):
        critic_par_count = 2 * args.n_var_layers * args.n_qubits
    elif args.quantum_critic and args.circuit == "Jerbi-reuploading":
        critic_par_count = 2 * (2 * args.n_enc_layers + 1) * args.n_qubits
    elif not args.quantum_critic:
        critic_par_count = (
            num_obs * args.critic_hidden_layer_nodes
            + args.critic_hidden_layer_nodes**2
            + args.critic_hidden_layer_nodes * 1
            + 2 * args.critic_hidden_layer_nodes
            + 1
        )

    if args.quantum_critic and args.hybrid:
        critic_par_count = critic_par_count + args.n_qubits * 1 + 1
    critic_par_count = critic_par_count * debug
    return critic_par_count


###############################################################
