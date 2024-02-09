from stable_baselines3 import DDPG ,PPO
from Policy import CustomActorCriticPolicy

from Environment import DemandForecastingEnv


def train_ddpg(timesteps=5e5):
    print("Created Environment...")
    env = DemandForecastingEnv()
    n_actions = env.action_space.shape[-1]
    agent = DDPG(CustomActorCriticPolicy, env, verbose=1)
    print("Starting Model Training...")
    agent.learn(total_timesteps=timesteps, log_interval=10)
    file_name = "./model/ddpg_model"
    agent.save(file_name)
    return agent

def train_ppo(timesteps=5e5):
    print("Created Environment...")
    env = DemandForecastingEnv()
    n_actions = env.action_space.shape[-1]
    agent = PPO(CustomActorCriticPolicy, env, verbose=1)
    print("Starting Model Training...")
    agent.learn(total_timesteps=timesteps, log_interval=10)
    file_name = "./model/ppo_model"
    agent.save(file_name)
    return agent

