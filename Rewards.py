import numpy as np

class RewardCalculator():
    def __init__(self): 
        print('initialize reward class ::: ')    

    def calculate_reward(self,true_val,predicted_val):          
        # Calculate the absolute difference between the true and the predicted values
        absolute_difference = [abs(test - predicted) for test, predicted in zip(true_val, predicted_val)]

        normalized_difference = (absolute_difference - np.min(absolute_difference)) / (np.max(absolute_difference) - np.min(absolute_difference))
        
        # Map the normalized difference to the desired reward range [10, 50] using a sigmoid function
        sigm_reward = 10 + normalized_difference

        reward = np.mean(sigm_reward)

        # Calculate the reward based on the difference
        return reward
  
        