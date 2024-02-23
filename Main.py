from Data_Preprocessing import preprocess
from Environment import DemandForecastingEnv
from sb3_contrib.ppo_recurrent import RecurrentPPO
from Test_Agent import testAgent
import numpy as np


input_window = 84
output_window = 14
interval = 28
preprocess_object = preprocess(input_window,output_window, interval)

## Train agent
train_x_seq, train_y_seq = preprocess_object.get_train_seq()
DF_Env = DemandForecastingEnv(train_x_seq,train_x_seq, 15)
n_actions = DF_Env.action_space.shape
agent = RecurrentPPO('MlpLstmPolicy', DF_Env, verbose = 1)
trained_agent = agent.learn(total_timesteps=100000, log_interval=10)
trained_agent.save("/models/recurrent_ppo_v2")


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true + 1), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# # ## Test agent
episodes = 1
model_path = ("C:/Users/priyanka.chakraborty/Projects/COE/RL/Projects/Fastenal_v2/Fastenal/models/recurrent_ppo_v2")
input_window = 12
output_window = 3
preprocess_object = preprocess(input_window,output_window)
test_x_seq, test_y_seq = preprocess_object.get_test_seq()
print("Test x seq dimensions:",test_x_seq.shape)
print("Test y seq dimensions:",test_y_seq.shape)
model = RecurrentPPO.load(model_path)
env = DemandForecastingEnv(test_x_seq, test_y_seq, 12)
done = False    
mean_absolute_error_list = []
obs = env.reset()


for i in range(episodes):
    action, states = model.predict(obs)
    mean_absolute_error_list.append(mean_absolute_percentage_error(test_y_seq,action))
    obs, reward, done, info = env.step(action)
    print("test_y_seq is::",test_y_seq)
    print("Mean absolute percentage error is::",mean_absolute_error_list)
    if done:
        env.reset()


