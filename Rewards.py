import numpy as np

class RewardCalculator():
    def __init__(self): 
        print('initialize reward class ::: ')    

    def calculate_reward(self,true_val,predicted_val):          
        # Calculate the absolute difference between the true and the predicted values
        difference = sum(abs(test - predicted) for test, predicted in zip(true_val, predicted_val))                                                                          
        # Calculate the reward based on the difference
        reward = 1.0 / (1.0 + difference)        
        return reward
        

  
        