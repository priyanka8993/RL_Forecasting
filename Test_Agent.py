from sb3_contrib.ppo_recurrent import RecurrentPPO
from Data_Preprocessing import preprocess
from Environment import DemandForecastingEnv
import pandas as pd 
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy


class testAgent():
    def __init__(self):
        self.model_path = ("C:/Users/priyanka.chakraborty/Projects/COE/RL/Projects/Fastenal_v2/Fastenal/models/recurrent_ppo")

    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def load_agent_model(self):
        print('file name --> ',self.model_path)
        agent = RecurrentPPO.load(self.model_path)
        return agent
    
    def evaluate_model(self,episodes):
        input_window = 12
        output_window = 3
        preprocess_object = preprocess(input_window,output_window)
        test_x_seq, test_y_seq = preprocess_object.get_test_seq()
        print("Test seq dimensions:",test_x_seq.shape)
        model = self.load_agent_model()
        env = DemandForecastingEnv(test_x_seq, test_y_seq, 12)
        done = False    
        # episode_rewards = []
        # eps_obs_list = []
 
        mean_absolute_error_list = []
        while not done:
            ''' return_episode_rewards=True to be removed from evaluate_policy if only mean reward per episode is required in the output '''
            # mean_reward, std_reward = evaluate_policy(model,env, n_eval_episodes=episodes)
            for i in range(episodes):
                obs = env.reset()
                action, states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                # total_reward += reward     
                # episode_rewards.append(reward) #N
                mean_absolute_error_list.append(self.mean_absolute_percentage_error(test_y_seq,action))
                # print('mean_reward : '+ str(mean_reward), 'std_reward : '+str(std_reward))
            # return mean_reward, std_reward
        return mean_absolute_error_list