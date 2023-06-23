
import numpy as np
from args import parse_args
args = parse_args()

def make_actor_layer_params():
    if (args.quantum_actor and args.data_re_uploading==False and (args.Jerbi_circuit==False or args.alt_circuit)):
        layer_params = np.random.rand(args.n_qubits, args.n_var_layers, 3)
        layer_params = layer_params*2 - 1                  #kann eventuell zu 0 werden
        layer_params = np.arctanh(layer_params)
    elif (args.quantum_actor and args.Jerbi_circuit and args.input_scaleing==False):
        layer_params = np.random.rand(args.n_qubits, args.n_var_layers, 2)
        layer_params = layer_params*2 - 1
        layer_params = np.arctanh(layer_params)
    #elif (args.quantum_actor and args.Jerbi_circuit and args.data_re_uploading==False):
    #    layer_params = np.random.rand(args.n_qubits, args.n_var_layers + 1, 2)
    #    layer_params = layer_params*2 - 1
    #    layer_params = np.arctanh(layer_params)
    elif (args.quantum_actor and args.Jerbi_circuit and args.data_re_uploading):
        layer_params = np.random.rand(args.n_qubits, (2 * args.n_enc_layers) + 1, 2)
        layer_params = layer_params*2 - 1
        layer_params = np.arctanh(layer_params)
        for i in range(args.n_qubits):
            for layer_nr in range(1, 2*args.n_enc_layers+1, 2):
                for k in range(2):
                        layer_params[i, layer_nr, k] = 1
    else:
        layer_params = np.random.rand(0)
        print("no actor_layer_params needed")
    return layer_params

def make_critic_layer_params():
    if (args.quantum_critic and args.data_re_uploading==False and (args.Jerbi_circuit==False or args.alt_circuit)):
        layer_params = np.random.rand(args.n_qubits, args.n_var_layers, 3)
        layer_params = layer_params*2 - 1                  #kann eventuell zu 0 werden
        layer_params = np.arctanh(layer_params)
    elif (args.quantum_critic and args.Jerbi_circuit and args.input_scaleing==False):
        layer_params = np.random.rand(args.n_qubits, args.n_var_layers, 2)
        layer_params = layer_params*2 - 1
        layer_params = np.arctanh(layer_params)
    elif (args.quantum_critic and args.data_re_uploading==True and args.Jerbi_circuit):
        layer_params = np.random.rand(args.n_qubits, (2 * args.n_enc_layers) + 1, 2)
        layer_params = layer_params*2 - 1
        layer_params = np.arctanh(layer_params)
        for i in range(args.n_qubits):
            for layer_nr in range(1, 2*args.n_enc_layers+1, 2):
                for k in range(2):
                        layer_params[i, layer_nr, k] = 1
    else:
        layer_params = np.random.rand(0)
        print("no critic_layer_params needed")
    return layer_params

