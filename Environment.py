
import gym
import numpy as np
from Rewards import RewardCalculator


class DemandForecastingEnv(gym.Env):

    def __init__(self, x_data,y_data, lookback_window):
        super(DemandForecastingEnv, self).__init__()
        self.x_data = x_data  # Time series data for demand forecasting
        self.y_data = y_data
        self.lookback_window = lookback_window  # Number of time steps to consider for forecasting
        self.reward_calc = RewardCalculator()
        self.current_step = 0

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(low=10.5, high=40.0 , shape=(7
                                                                        ,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=10.0, high=50.0, shape=(30,)) 

    def reset(self):
        self.current_step = 0
        initial_observation = self.x_data[self.current_step]
        return initial_observation

    def step(self, action):
        self.current_step += 1
        # Execute the action and move the time step forward
        next_state = self.x_data[self.current_step]
        # Calculate reward
        reward = self.calculate_reward(action, next_state)
        # Check if the episode is done (end of data)
        done = self.current_step >= len(self.x_data) - self.lookback_window
        print("action is :::", action)
        print("reward is :::", reward)
        print("state is :::", next_state)
        return next_state, reward, done, {}
        # elif self.current_step == 0:
        #     next_state = self.x_data[self.current_step]
        #     # Calculate reward
        #     reward = self.calculate_reward(action, next_state)
        #     # Check if the episode is done (end of data)
        #     done = self.current_step >= len(self.x_data) - self.lookback_window
        #     return next_state, reward, done, {}


    def calculate_reward(self, action, observation):
        current_reward =self.reward_calc.calculate_reward(observation,action)
        return current_reward

 
