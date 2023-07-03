import os
import pandas as pd
import torch.nn as nn


class Save_results(nn.Module):
    def __init__(self, results_dir, load_chkpt):
        super(Save_results, self).__init__()
        self.results_dir = results_dir

        self.dict_episode = {
                            'episode_reward': [],
                            'episode_length': [],

                            'global_step': [],
                            'gym_id': [],
                            'exp_name': [],
                            'circuit': [],
                            'seed': []
        }
        self.dict_update = {
                            'learning_rate': [],
                            'qlearning_rate': [],
                            'value_loss': [],
                            'policy_loss': [],
                            'entropy': [],
                            'loss': [],
                            'old_approx_kl': [],
                            'approx_kl': [],
                            'clipfrac': [],
                            'explained_variance': [],
                            'SPS': [],
                            'output_scaleing': [],

                            'global_step': [],
                            'gym_id': [],
                            'exp_name': [],
                            'circuit': [],
                            'seed': []
        }
        file_pathExists = os.path.exists(self.results_dir)
        if not file_pathExists:
            os.makedirs(self.results_dir)
            if load_chkpt:
                df_pathExists = os.path.exists(self.results_dir + "/episode_results.csv")
                if not df_pathExists:
                    print("ERROR can not append results")
                else:
                    self.dict_episode.update(pd.read_csv(self.results_dir + "/episode_results.csv").to_dict('list'))
                    self.dict_update.update(pd.read_csv(self.results_dir + "/update_results.csv").to_dict('list'))
                    

    def append_episode_results(self, episode_reward, episode_length, global_step, gym_id, exp_name, circuit, seed):
        #append episode statistics
        self.dict_episode['episode_reward'].append(episode_reward)
        self.dict_episode['episode_length'].append(episode_length)
        #append meta information obout the Algorithm
        self.dict_episode['global_step'].append(global_step)
        self.dict_episode['gym_id'].append(gym_id)
        self.dict_episode['exp_name'].append(exp_name)
        self.dict_episode['circuit'].append(circuit)
        self.dict_episode['seed'].append(seed)

        
    def append_update_results(self, learning_rate, qlearning_rate, value_loss, policy_loss, entropy, loss, old_approx_kl, approx_kl, clipfrac, explained_variance, SPS, output_scaleing, global_step, gym_id, exp_name, circuit, seed):
        #append update statistics
        self.dict_update['learning_rate'].append(learning_rate)
        self.dict_update['qlearning_rate'].append(qlearning_rate)
        self.dict_update['value_loss'].append(value_loss)
        self.dict_update['policy_loss'].append(policy_loss)
        self.dict_update['entropy'].append(entropy)
        self.dict_update['loss'].append(loss)
        self.dict_update['old_approx_kl'].append(old_approx_kl)
        self.dict_update['approx_kl'].append(approx_kl)
        self.dict_update['clipfrac'].append(clipfrac)
        self.dict_update['explained_variance'].append(explained_variance)
        self.dict_update['SPS'].append(SPS)
        self.dict_update['output_scaleing'].append(output_scaleing)
        #append meta information obout the Algorithm
        self.dict_update['global_step'].append(global_step)
        self.dict_update['gym_id'].append(gym_id)
        self.dict_update['exp_name'].append(exp_name)
        self.dict_update['circuit'].append(circuit)
        self.dict_update['seed'].append(seed)

    def save_results(self):
        df_episode = pd.DataFrame(data=self.dict_episode)
        df_episode.to_csv(self.results_dir + "/episode_results.csv", sep=",", encoding="utf-8", index=False)
    
        df_update = pd.DataFrame(data=self.dict_update) #, index=[0]
        df_update.to_csv(self.results_dir + "/update_results.csv", sep=",", encoding="utf-8", index=False)



