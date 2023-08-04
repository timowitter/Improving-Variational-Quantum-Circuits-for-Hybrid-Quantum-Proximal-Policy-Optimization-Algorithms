import math

import numpy as np

from args import parse_args

args = parse_args()


def make_random_init(n_qubits, n_layers, n_dims):
    layer_params = np.random.rand(n_qubits, n_layers, n_dims)
    layer_params = layer_params * 2 - 1  # could be set to 0
    layer_params = np.arctanh(layer_params)  # will later be inside a tanh function
    return layer_params


a = 0.01  # could be made a hyperparameter but is probaly uninteresting for tests


def make_clipped_random_init(n_qubits, n_layers, n_dims):
    layer_params = np.random.rand(n_qubits, n_layers, n_dims)
    layer_params = layer_params * 2 - 1  # could be set to 0
    # clipping to avoid vanishing gradients
    for i in range(n_qubits):
        for j in range(n_layers):
            for k in range(n_dims):
                if layer_params[i, j, k] >= 0:
                    layer_params[i, j, k] = np.clip(layer_params[i, j, k], 0 + a, 1 - a)
                else:
                    layer_params[i, j, k] = np.clip(layer_params[i, j, k], -1 + a, 0 - a)
    layer_params = np.arctanh(layer_params)  # will later be inside a tanh function
    return layer_params


def make_gauss_init(n_qubits, n_layers, n_dims):
    layer_params = np.random.normal(0, 1, (n_qubits, n_layers, n_dims))
    # could be initialised at 0 so we clip it to avoid vanishing gradients
    for i in range(n_qubits):
        for j in range(n_layers):
            for k in range(n_dims):
                if layer_params[i, j, k] >= 0:
                    layer_params[i, j, k] = np.clip(
                        layer_params[i, j, k], 0 + a, np.arctanh(1 - a)
                    )
                else:
                    layer_params[i, j, k] = np.clip(
                        layer_params[i, j, k], np.arctanh(-1 + a), 0 - a
                    )
    return layer_params


def make_actor_layer_params():
    if args.quantum_actor and (
        args.circuit == "simple"
        or args.circuit == "Hgog"
        or args.circuit == "simple_reuploading"
        or args.circuit == "Hgog_reuploading"
    ):
        layer_params = make_random_init(args.n_qubits, args.n_var_layers, 3)

    elif args.quantum_actor and (
        args.circuit == "simple_reuploading_with_input_scaleing"
        or args.circuit == "Hgog_reuploading_with_output_scaleing"
    ):
        # enc layers have onely one dim per layer (1 enc layer per var layer)
        num_enc_dims = math.ceil(args.n_var_layers / 3)
        # we put them in the same array, so we dont need an additional optimizer
        layer_params = make_random_init(args.n_qubits, args.n_var_layers + num_enc_dims, 3)
        # set input scaleing to 1 at start
        for i in range(args.n_qubits):
            for layer_nr in range(args.n_var_layers, args.n_var_layers + num_enc_dims):
                for k in range(3):
                    layer_params[i, layer_nr, k] = 1

    elif args.quantum_actor and (
        args.circuit == "Jerbi-no-reuploading-no-input-scaleing"
        or args.circuit == "Jerbi-reuploading-no-input-scaleing"
    ):
        layer_params = make_random_init(args.n_qubits, args.n_var_layers, 2)

    elif args.quantum_actor and args.circuit == "Jerbi-reuploading":
        layer_params = make_random_init(args.n_qubits, (2 * args.n_enc_layers) + 1, 2)
        # set input scaleing to 1 at start
        for i in range(args.n_qubits):
            for layer_nr in range(1, 2 * args.n_enc_layers + 1, 2):
                for k in range(2):
                    layer_params[i, layer_nr, k] = 1
    else:
        layer_params = np.random.rand(0)
        print("no actor_layer_params needed")
    return layer_params


def make_critic_layer_params():
    if args.quantum_actor and (
        args.circuit == "simple"
        or args.circuit == "Hgog"
        or args.circuit == "simple_reuploading"
        or args.circuit == "Hgog_reuploading"
    ):
        layer_params = make_random_init(args.n_qubits, args.n_var_layers, 3)

    elif args.quantum_actor and (
        args.circuit == "simple_reuploading_with_input_scaleing"
        or args.circuit == "Hgog_reuploading_with_output_scaleing"
    ):
        # enc layers have onely one dim per layer (1 enc layer per var layer)
        num_enc_dims = math.ceil(args.n_var_layers / 3)
        # we put them in the same array, so we dont need an additional optimizer
        layer_params = make_random_init(args.n_qubits, args.n_var_layers + num_enc_dims, 3)
        # set input scaleing to 1 at start
        for i in range(args.n_qubits):
            for layer_nr in range(args.n_var_layers, args.n_var_layers + num_enc_dims):
                for k in range(3):
                    layer_params[i, layer_nr, k] = 1

    elif args.quantum_critic and (
        args.circuit == "Jerbi-no-reuploading-no-input-scaleing"
        or args.circuit == "Jerbi-reuploading-no-input-scaleing"
    ):
        layer_params = make_random_init(args.n_qubits, args.n_var_layers, 2)

    elif args.quantum_critic and args.circuit == "Jerbi-reuploading":
        layer_params = make_random_init(args.n_qubits, (2 * args.n_enc_layers) + 1, 2)
        # set input scaleing to 1 at start
        for i in range(args.n_qubits):
            for layer_nr in range(1, 2 * args.n_enc_layers + 1, 2):
                for k in range(2):
                    layer_params[i, layer_nr, k] = 1

    else:
        layer_params = np.random.rand(0)
        print("no critic_layer_params needed")
    return layer_params
