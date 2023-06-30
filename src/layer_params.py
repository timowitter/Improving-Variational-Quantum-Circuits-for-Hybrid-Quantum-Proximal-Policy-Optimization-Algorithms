
import numpy as np
from args import parse_args
args = parse_args()

def make_actor_layer_params():
    if (args.quantum_actor and (args.circuit=="simple" or args.circuit=="Hgog")):
        layer_params = np.random.rand(args.n_qubits, args.n_var_layers, 3)
        layer_params = layer_params*2 - 1                  #kann eventuell zu 0 werden
        layer_params = np.arctanh(layer_params)
    elif (args.quantum_actor and (args.circuit=="Jerbi-no-reuploading-no-input-scaleing" or args.circuit=="Jerbi-reuploading-no-input-scaleing")):
        layer_params = np.random.rand(args.n_qubits, args.n_var_layers, 2)
        layer_params = layer_params*2 - 1
        layer_params = np.arctanh(layer_params)
    elif (args.quantum_actor and args.circuit=="Jerbi-reuploading"):
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
    if (args.quantum_critic and (args.circuit=="simple" or args.circuit=="Hgog")):
        layer_params = np.random.rand(args.n_qubits, args.n_var_layers, 3)
        layer_params = layer_params*2 - 1                  #kann eventuell zu 0 werden
        layer_params = np.arctanh(layer_params)
    elif (args.quantum_critic and (args.circuit=="Jerbi-no-reuploading-no-input-scaleing" or args.circuit=="Jerbi-reuploading-no-input-scaleing")):
        layer_params = np.random.rand(args.n_qubits, args.n_var_layers, 2)
        layer_params = layer_params*2 - 1
        layer_params = np.arctanh(layer_params)
    elif (args.quantum_critic and args.circuit=="Jerbi-reuploading"):
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

