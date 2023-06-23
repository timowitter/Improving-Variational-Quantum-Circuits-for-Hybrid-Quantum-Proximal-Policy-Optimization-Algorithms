import argparse
import os
from distutils.util import strtobool
import json

def save_args(args):
    args_chkpt_file = os.path.join(args.chkpt_dir, 'commandline_args.txt')
    #if os.path.exists(args_chkpt_file): # delete old file with same name if it exists and make a new run
    #        os.remove(args_chkpt_file)
    with open(args_chkpt_file, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

"""
def load_args(args):
    t=args.total_timesteps    
    args_chkpt_file = os.path.join(args.chkpt_dir, 'commandline_args.txt')
    with open(args_chkpt_file, 'r') as f:
        args.__dict__ = json.load(f)
    args.load_chkpt = True
    args.total_timesteps = t"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"), help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="CartPole-v1", help='id of the OpenAi Gym Environment /default: CartPole-v1')
    parser.add_argument('--warmup-learning-rate-bonus', type=float, default=2.5e-4, help='optimizer warmup learning rate')                                             
    parser.add_argument('--warmup-qlearning-rate-bonus', type=float, default=5e-3, help='quantum optimizer warmup learning rate')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4, help='optimizer learning rate')                                                                 #2.5e-4
    parser.add_argument('--qlearning-rate', type=float, default=1e-3, help='quantum optimizer learning rate')
    parser.add_argument('--output-scaleing-learning-rate', type=float, default=1.5e-4, help='output scaleing learning rate')
    parser.add_argument('--output-scaleing-start', type=float, default=1.0, help='output scaleing start value')
    parser.add_argument('--seed', type=int, default=1, help='seed for random events')
    parser.add_argument('--total-timesteps', type=int, default=50000, help='length of the Training')                                                                    #25000
    parser.add_argument('--warmup-timesteps', type=int, default=50000, help='length of the warmup Phase') 
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='cdnn.deterministic use /default:True')
    #parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True, help='determines if gpu shall be used /default:False')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True, help='wether to record videos to `videos` folder')   #not working

    #Algorith specific
    parser.add_argument('--num-envs', type=int, default=1, help='number of environments run in parrallel in the SyncVectorEnv')                                         #1 Vector environment #4
    parser.add_argument('--num-steps', type=int, default=128, help='the number of steps in each environments per policy rollout phase of the multy vector env')
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True, help='leaning rate annealing for ploicy and value networks')  #4 lr Annealing
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='GAE for Advantage computation')                           #5 general advantage estimation
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor gamma')                                                                              #0.98
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='lsmbda for general advantage estimation')
    parser.add_argument('--num-minibatches', type=int, default=4, help='number of minibatches')                                                                         #6 minibatches
    parser.add_argument('--update-epochs', type=int, default=4, help='the k epochs to update the policy')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='toggle Advantage normalisation')                     #7 Advatage Normalisation
    parser.add_argument('--clip-coef', type=float, default=0.2, help='clipping coefficient')                                                                            #8 clipped Objektive
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='toggle the use of clipped value loss')             #9 Value loss clipping
    parser.add_argument('--ent-coef', type=float, default=0.00, help='coefficient for the entropy (loss), 0 = disabled')                                                #10 entropy loss #0.01
    parser.add_argument('--vf-coef', type=float, default=0.5, help='coefficient for the value function')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='maximum norm for the gradient clipping')                                                      #11 global gradient clipping
    parser.add_argument('--target-kl', type=float, default=None, help='the target KL divergence threshhold (0.015)')                                                    #target KL early stopping


    parser.add_argument('--actor-hidden-layer-nodes', type=int, default=64, help='number of nodes of the 2 hidden layers of the actor')
    parser.add_argument('--critic-hidden-layer-nodes', type=int, default=64, help='number of nodes of the 2 hidden layers of the critic')
    parser.add_argument('--quantum-actor', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='if True quantum actor will be used')
    parser.add_argument('--quantum-critic', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True, help='if True quantum critic will be used')
    parser.add_argument('--n-qubits', type=int, default=4, help='number of qubits of the circuit')
    parser.add_argument('--n-var-layers', type=int, default=2, help='gives the number of variational layers in case of the simple circuit')
    parser.add_argument('--n-enc-layers', type=int, default=1, help='gives the the number of encodeing layers in case of data reuploading (then the number of variational layers equals the number of encodeing layers + 1)')
    parser.add_argument('--alt-circuit', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True, help='toggle alternate simple circuit by Mohamad Hgog')
    parser.add_argument('--Jerbi-circuit', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='toggle alternate simple circuit by Jerbi et al.')  
    parser.add_argument('--data-re-uploading', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='toggle data-re-uploading') 
    parser.add_argument('--input-scaleing', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='toggle input scaleing') 
    parser.add_argument('--hybrid', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True, help='toggle hybrid output postprocessing') 
    parser.add_argument('--epsylon-greedy', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=False, help='epsylon greedy action selection instead of sampling')
    parser.add_argument('--epsylon', type=float, default=1.0, help='random action selection chance at start')   
    parser.add_argument('--load-chkpt', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=False, help='continue learning from checkpoint')
    parser.add_argument('--save-location', type=str, default='tmp/qppo', help='save-location/gym-id/exp-name/seed make the save location of the experiment (chkpt-dir)')
    parser.add_argument('--save-intervall', type=int, default=4, help='gives the the save-intervall in number of update epochs')


    args=parser.parse_args()
    args.batch_size=int(args.num_steps * args.num_envs)
    args.minibatch_size=int(args.batch_size // args.num_minibatches)
    args.chkpt_dir = f"{args.save_location}/{args.gym_id}/{args.exp_name}/{args.seed}"

    

    if (args.load_chkpt):                   # load args exept total_timesteps and continue learning if new total_timesteps > global_timesteps
        args_chkpt_file = os.path.join(args.chkpt_dir, 'commandline_args.txt')
        t=args.total_timesteps    
        with open(args_chkpt_file, 'r') as f:
            args.__dict__ = json.load(f)
        args.load_chkpt = True
        args.total_timesteps = t

    return args