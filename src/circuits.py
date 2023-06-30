import numpy as np
import torch
import pennylane as qml

from transform_funks import transform_obs_to_binary, normalize_obs
from args import parse_args
args = parse_args()

#######################################################################################################################################################
##                                              quantum actor circuits                                                                               ##
##                                                                                                                                                   ##
dev = qml.device("default.qubit", wires=args.n_qubits)
#simple circuit:            (by Yunseok Kwak et all in Introduction to Quantum Reinforcement Learning: Theory and PennyLane-based Implementation)

#Parameterized Rotation & Entanglement Layers
def simple_layer(layer_params, layer_nr):
    for i in range(args.n_qubits):
        qml.RX(layer_params[i, layer_nr, 0], wires=i)
        qml.RY(layer_params[i, layer_nr, 1], wires=i)
        qml.RZ(layer_params[i, layer_nr, 2], wires=i)

    if (((layer_nr == args.n_var_layers-1) and (args.gym_id == "CartPole-v0" or args.gym_id == "CartPole-v1") and args.n_qubits == 4)):
        qml.CNOT(wires=[0,2])
        qml.CNOT(wires=[1,3])
        #for i in range((args.n_qubits - act_dim), args.n_qubits):
        #    qml.CNOT(wires=[i-2,i])
    else:
        for i in range(args.n_qubits-1):
            qml.CNOT(wires=[i,i+1])
    #qml.CNOT(wires=[args.n_qubits-1, 0])

#Variational Quantum Policy Circuit (Actor)
@qml.qnode(dev, interface='torch', diff_method="backprop")
def simple_actor_circuit(layer_params, observation, act_dim):
    norm_obs = normalize_obs(observation, args.gym_id, args.n_qubits)

    # Input Encoding
    for i in range(args.n_qubits):
        qml.RY(np.pi*norm_obs[i], wires=i)

    # Variational Quantum Circuit
    for layer_nr in range(args.n_var_layers):
            simple_layer(np.pi*torch.tanh(layer_params), layer_nr)

    if(args.hybrid):
        return [qml.expval(qml.PauliZ(ind)) for ind in range(args.n_qubits)]
    else:    
        return [qml.expval(qml.PauliZ(ind)) for ind in range((args.n_qubits - act_dim), args.n_qubits)]



#alternative simple circuit:            (by Mohamad Hgog in: Quantum-Enhanced Policy Gradient Methods for Reinforcement Learning)

#Parameterized Rotation & Entanglement Layers
def Hgog_layer(layer_params, layer_nr):
    for i in range(args.n_qubits):
        qml.RZ(layer_params[i, layer_nr, 0], wires=i)
        qml.RY(layer_params[i, layer_nr, 1], wires=i)
        qml.RZ(layer_params[i, layer_nr, 2], wires=i)

    for i in range(args.n_qubits-1):
        qml.CNOT(wires=[i,i+1])
    #qml.CNOT(wires=[args.n_qubits-1, 0])

#Variational Quantum Policy Circuit (Actor)
@qml.qnode(dev, interface='torch', diff_method="backprop")
def Hgog_actor_circuit(layer_params, observation, act_dim):
    # layer_params: Variable Layer Parameters, observation: State Variable
    norm_obs = normalize_obs(observation, args.gym_id, args.n_qubits)
    # Input Encoding
    for i in range(args.n_qubits):
        qml.RX(np.pi*norm_obs[i], wires=i)

    # Variational Quantum Circuit
    for layer_nr in range(args.n_var_layers):
            Hgog_layer(np.pi*torch.tanh(layer_params), layer_nr)

    if(args.hybrid):
        return [qml.expval(qml.PauliZ(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliZ(ind)) for ind in range(act_dim)]


#data reuploading circuit:             (by Jerbi et al. in: Parametrized Quantum Policies for Reinforcement Learning)

def variational_layer(layer_params, layer_nr):
    for i in range(args.n_qubits):
        qml.RZ(np.pi*torch.tanh(layer_params[i, layer_nr, 0]), wires=i)
        qml.RY(np.pi*torch.tanh(layer_params[i, layer_nr, 1]), wires=i)

    for i in range(args.n_qubits-1):
        qml.CNOT(wires=[i,i+1])
    qml.CNOT(wires=[args.n_qubits-1, 0])

def encodeing_layer(encodeing_params, layer_nr, state_vector):
    for i in range(args.n_qubits):
        qml.RY(np.pi*torch.tanh(encodeing_params[i, layer_nr, 0] * state_vector[i]), wires=i)
        qml.RZ(np.pi*torch.tanh(encodeing_params[i, layer_nr, 1] * state_vector[i]), wires=i)

#Variational Quantum Policy Circuit (Actor)
@qml.qnode(dev, interface='torch', diff_method="backprop")
def Jerbi_reuploading_actor_circuit(layer_params, observation, act_dim):
    # layer_params: Variable Layer Parameters, observation: State Variable
    for i in range(args.n_qubits):
        qml.Hadamard(wires=i)

    variational_layer(layer_params, 0)
    
    for layer_nr in range(1, 2*args.n_enc_layers+1, 2):
        if (args.gym_id == "FrozenLake-v0" or args.gym_id == "FrozenLake-v1" or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"):
            encodeing_layer(layer_params, layer_nr, transform_obs_to_binary(observation, args.n_qubits))
        else:
            encodeing_layer(layer_params, layer_nr, observation)
        variational_layer(layer_params, layer_nr+1)

    if(args.hybrid):
        return [qml.expval(qml.PauliY(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliY(ind)) for ind in range(act_dim)]


#Variational Quantum Policy Circuit (Actor)
@qml.qnode(dev, interface='torch', diff_method="backprop")
def Jerbi_actor_circuit_no_reuploading_no_input_scaleing(layer_params, observation, act_dim):
    norm_obs = normalize_obs(observation, args.gym_id, args.n_qubits)
    # layer_params: Variable Layer Parameters, observation: State Variable
    for i in range(args.n_qubits):
        qml.Hadamard(wires=i)

    variational_layer(layer_params, 0)

    for i in range(args.n_qubits):
        qml.RY(np.pi*norm_obs[i], wires=i)
        qml.RZ(np.pi*norm_obs[i], wires=i)
    
    for layer_nr in range(1, args.n_var_layers):
        variational_layer(layer_params, layer_nr)

    if(args.hybrid):
        return [qml.expval(qml.PauliY(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliY(ind)) for ind in range(act_dim)]
    

#Variational Quantum Policy Circuit (Actor)
@qml.qnode(dev, interface='torch', diff_method="backprop")
def Jerbi_reuploading_actor_circuit_without_input_scaleing(layer_params, observation, act_dim):
    norm_obs = normalize_obs(observation, args.gym_id, args.n_qubits)
    # layer_params: Variable Layer Parameters, observation: State Variable
    for i in range(args.n_qubits):
        qml.Hadamard(wires=i)

    variational_layer(layer_params, 0)
    
    for layer_nr in range(1, args.n_var_layers):
        for i in range(args.n_qubits):
            qml.RY(np.pi*norm_obs[i], wires=i)
            qml.RZ(np.pi*norm_obs[i], wires=i)
        variational_layer(layer_params, layer_nr)

    if(args.hybrid):
        return [qml.expval(qml.PauliY(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliY(ind)) for ind in range(act_dim)]



@qml.qnode(dev, interface='torch', diff_method="backprop")
def empty_circuit(layer_params, observation, act_dim=None):
    return
##                                                                                                                                                   ##
##                                              quantum actor circuits                                                                               ##
#######################################################################################################################################################


def actor_circuit_selection():
    #circuit selection
    if (args.quantum_actor and args.circuit=="simple"):
        actor_circuit = simple_actor_circuit
        print("useing simple Quantum Circuit as actor")
    elif (args.quantum_actor and args.circuit=="Hgog"):
        actor_circuit = Hgog_actor_circuit
        print("useing alternative simple Quantum Circuit as actor")
    elif (args.quantum_actor and args.circuit=="Jerbi-no-reuploading-no-input-scaleing"):
        actor_circuit = Jerbi_actor_circuit_no_reuploading_no_input_scaleing
        print("useing Jerbi simple Quantum Circuit as actor")
    elif (args.quantum_actor and args.circuit=="Jerbi-reuploading"):
        actor_circuit = Jerbi_reuploading_actor_circuit
        print("useing Jerbi data reuploading Quantum Circuit as actor")
    elif (args.quantum_actor and args.circuit=="Jerbi-reuploading-no-input-scaleing"):
        actor_circuit = Jerbi_reuploading_actor_circuit_without_input_scaleing
        print("useing Jerbi reuploading Quantum Circuit without output scaleing as actor")
    elif (args.quantum_actor==False):
        actor_circuit = empty_circuit
        print("useing classical actor")
    else:
        print("actor selection ERROR")

    if (args.quantum_actor and args.hybrid):
        print("with hybrid output postprocessing")
    elif(args.quantum_actor and args.hybrid==False):
        print("with output scaleing")
    else:
        print("without output scaleing")
    
    return actor_circuit




#######################################################################################################################################################
##                                              quantum circuits                                                                                     ##
##                                                                                                                                                   ##
#simple circuit:            (by Yunseok Kwak et all in Introduction to Quantum Reinforcement Learning: Theory and PennyLane-based Implementation)

#Variational Quantum Policy Circuit (Actor)
@qml.qnode(dev, interface='torch', diff_method="backprop")
def simple_critic_circuit(layer_params, observation):
    norm_obs = normalize_obs(observation)

    # Input Encoding
    for i in range(args.n_qubits):
        qml.RY(np.pi*norm_obs[i], wires=i)

    # Variational Quantum Circuit
    for layer_nr in range(args.n_var_layers):
            simple_layer(np.pi*torch.tanh(layer_params), layer_nr)

    if(args.hybrid):
        return [qml.expval(qml.PauliZ(ind)) for ind in range(args.n_qubits)]
    else:    
        return [qml.expval(qml.PauliZ(args.n_qubits - 1))]



#alternative simple circuit:            (by Mohamad Hgog in: Quantum-Enhanced Policy Gradient Methods for Reinforcement Learning)

#Variational Quantum Policy Circuit (Actor)
@qml.qnode(dev, interface='torch', diff_method="backprop")
def Hgog_critic_circuit(layer_params, observation):
    # layer_params: Variable Layer Parameters, observation: State Variable
    norm_obs = normalize_obs(observation)
    # Input Encoding
    for i in range(args.n_qubits):
        qml.RX(np.pi*norm_obs[i], wires=i)

    # Variational Quantum Circuit
    for layer_nr in range(args.n_var_layers):
            Hgog_layer(np.pi*torch.tanh(layer_params), layer_nr)

    if(args.hybrid):
        return [qml.expval(qml.PauliZ(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliZ(args.n_qubits - 1))]


#data reuploading circuit:

#Variational Quantum Policy Circuit (Actor)
@qml.qnode(dev, interface='torch', diff_method="backprop")
def Jerbi_reuploading_critic_circuit(layer_params, observation):
    # layer_params: Variable Layer Parameters, observation: State Variable
    for i in range(args.n_qubits):
        qml.Hadamard(wires=i)

    variational_layer(layer_params, 0)

    if (args.gym_id == "FrozenLake-v0" or args.gym_id == "FrozenLake-v1" or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"):
        for layer_nr in range(1, 2*args.n_enc_layers+1, 2):
            encodeing_layer(layer_params, layer_nr, transform_obs_to_binary(observation))
            variational_layer(layer_params, layer_nr+1)
    else:
        for layer_nr in range(1, 2*args.n_enc_layers+1, 2):
            encodeing_layer(layer_params, layer_nr, observation)
            variational_layer(layer_params, layer_nr+1)

    if(args.hybrid):
        return [qml.expval(qml.PauliY(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliY(args.n_qubits - 1))]


#Variational Quantum Policy Circuit (Actor)
@qml.qnode(dev, interface='torch', diff_method="backprop")
def Jerbi_critic_circuit_no_reuploading_no_input_scaleing(layer_params, observation):
    norm_obs = normalize_obs(observation)
    # layer_params: Variable Layer Parameters, observation: State Variable
    for i in range(args.n_qubits):
        qml.Hadamard(wires=i)

    variational_layer(layer_params, 0)

    for i in range(args.n_qubits):
        qml.RY(np.pi*norm_obs[i], wires=i)
        qml.RZ(np.pi*norm_obs[i], wires=i)
    
    for layer_nr in range(1, args.n_var_layers):
        variational_layer(layer_params, layer_nr)

    if(args.hybrid):
        return [qml.expval(qml.PauliY(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliY(args.n_qubits - 1))]
    

@qml.qnode(dev, interface='torch', diff_method="backprop")
def Jerbi_reuploading_critic_circuit_without_input_scaleing(layer_params, observation):
    norm_obs = normalize_obs(observation)
    # layer_params: Variable Layer Parameters, observation: State Variable
    for i in range(args.n_qubits):
        qml.Hadamard(wires=i)

    variational_layer(layer_params, 0)
    
    for layer_nr in range(1, args.n_var_layers):
        for i in range(args.n_qubits):
            qml.RY(np.pi*norm_obs[i], wires=i)
            qml.RZ(np.pi*norm_obs[i], wires=i)
        variational_layer(layer_params, layer_nr)

    if(args.hybrid):
        return [qml.expval(qml.PauliY(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliY(args.n_qubits - 1))]
##                                                                                                                                                   ##
##                                              quantum critic circuits                                                                              ##
#######################################################################################################################################################


def critic_circuit_selection():
    if (args.quantum_critic and args.circuit=="simple"):
        critic_circuit = simple_critic_circuit
        print("useing simple Quantum Circuit as actor")
    elif (args.quantum_critic and args.circuit=="Hgog"):
        critic_circuit = Hgog_critic_circuit
        print("useing alternative simple Quantum Circuit as actor")
    elif (args.quantum_critic and args.circuit=="Jerbi-no-reuploading-no-input-scaleing"):
        critic_circuit = Jerbi_critic_circuit_no_reuploading_no_input_scaleing
        print("useing Jerbi simple Quantum Circuit as actor")
    elif (args.quantum_critic and args.circuit=="Jerbi-reuploading"):
        critic_circuit = Jerbi_reuploading_critic_circuit
        print("useing Jerbi data reuploading Quantum Circuit as actor")
    elif (args.quantum_critic and args.circuit=="Jerbi-reuploading-no-input-scaleing"):
        critic_circuit = Jerbi_reuploading_critic_circuit_without_input_scaleing
        print("useing Jerbi reuploading Quantum Circuit without output scaleing as actor")
    elif (args.quantum_critic==False):
        critic_circuit = empty_circuit
        print("useing classical actor")
    else:
        print("actor selection ERROR")

    if (args.quantum_critic and args.hybrid):
        print("with hybrid output postprocessing")
    elif(args.quantum_critic and args.hybrid==False):
        print("with manual output rescaleing")
    else:
        print("without output scaleing")

    return critic_circuit
    

    


