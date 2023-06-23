from args import parse_args
args = parse_args()

#number of parameters for a neural network layer: number_of_inputs * number_of_outputs + number_of_outputs
#                                                 (             weights              ) + (     biases    ) 

#number of parameters for a quantum circuit:      layers * dimensions   +   (output scaleing   or   hybrid neural network)

#calculate number of trainable actor parameters (for run_name)
###############################################################
def calc_num_actor_params():
    actor_par_count = 0
    if (args.gym_id == "FrozenLake-v0" or args.gym_id == "FrozenLake-v1" or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"):
        num_obs=16; num_acts=4;  debug=1
    elif (args.gym_id == "CartPole-v0" or args.gym_id == "CartPole-v1"):
        num_obs=4; num_acts=2;   debug=1
    else:
        num_obs=0; num_acts=0;   debug=0
        print("number of parameter calculation error")
    
    if (args.quantum_actor and args.data_re_uploading==False and (args.Jerbi_circuit==False or args.alt_circuit)):
        actor_par_count = 3*args.n_var_layers*args.n_qubits
    elif (args.quantum_actor and args.Jerbi_circuit and args.input_scaleing==False):
        actor_par_count = 2*args.n_var_layers*args.n_qubits
    elif (args.quantum_actor and args.data_re_uploading==True and args.Jerbi_circuit):
        actor_par_count = 2*(2*args.n_enc_layers+1)*args.n_qubits
    elif (args.quantum_actor==False):
        actor_par_count = num_obs*args.actor_hidden_layer_nodes + args.actor_hidden_layer_nodes**2 + args.actor_hidden_layer_nodes*num_acts + 2*args.actor_hidden_layer_nodes + num_acts

    if (args.quantum_actor and args.hybrid):
        actor_par_count = actor_par_count + args.n_qubits*num_acts + num_acts
    elif(args.quantum_actor and args.hybrid==False):
        actor_par_count = actor_par_count + num_acts
    actor_par_count = actor_par_count*debug
    return actor_par_count
###############################################################


#calculate number of trainable critic parameters (for run_name)
###############################################################
def calc_num_critic_params():
    critic_par_count = 0
    if (args.gym_id == "FrozenLake-v0" or args.gym_id == "FrozenLake-v1" or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"):
        num_obs=16; num_acts=4;  debug=1
    elif (args.gym_id == "CartPole-v0" or args.gym_id == "CartPole-v1"):
        num_obs=4; num_acts=2;   debug=1
    else:
        num_obs=0; num_acts=0;   debug=0
        print("number of parameter calculation error")

    if (args.quantum_critic and args.data_re_uploading==False and (args.Jerbi_circuit==False or args.alt_circuit)):
        critic_par_count = 3*args.n_var_layers*args.n_qubits
    elif (args.quantum_critic and args.Jerbi_circuit and args.input_scaleing==False):
        critic_par_count = 2*args.n_var_layers*args.n_qubits
    elif (args.quantum_critic and args.data_re_uploading==True and args.Jerbi_circuit):
        critic_par_count = 2*(2*args.n_enc_layers+1)*args.n_qubits
    elif (args.quantum_critic==False):
        critic_par_count = num_obs*args.critic_hidden_layer_nodes + args.critic_hidden_layer_nodes**2 + args.critic_hidden_layer_nodes*1 + 2*args.critic_hidden_layer_nodes + 1

    if (args.quantum_critic and args.hybrid):
        critic_par_count = critic_par_count + args.n_qubits*1 + 1
    critic_par_count = critic_par_count*debug
    return critic_par_count
###############################################################