
# Usage example:
from Environment import DemandForecastingEnv
import stable_baselines3 
from stable_baselines3 import DDPG,PPO
import pandas as pd 


class test():
    def __init__(self) :
        test_sequence = self.test_sequence
        model_path = self.model_path

    def load_model(self):
        agent = DDPG.load(self.model_path)
        return agent 

    def test(self):
        agent = self.load_model()
        env = DemandForecastingEnv(self.test_sequence)

        done = False
        total_reward = 0
        time_step = 0 
        while not done:
            action ,_states = agent.predict(env)
            obs, reward, done, info = env.step(action)
            total_reward += reward 
            time_step += 1

        print("Reward",total_reward)
        print("Timesteps",time_step)