from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces
import numpy as np

import torch as th
from torch import nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = th.device('cpu')
if(th.cuda.is_available()): 
    device = th.device('cuda:0') 
    th.cuda.empty_cache()
    print("Device set to : " + str(th.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

INPUT_SEQUENCE =12
OUTPUT_SEQUENCE=3
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):

        super(ActorNetwork, self).__init__()      
        
        self.bilstm = nn.LSTM(input_size=input_dim, hidden_size=32, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):

        x, _ = self.bilstm(x)     
        x = self.dropout(x)        
        x = self.fc(x)
       
        return x
    
class CriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CriticNetwork, self).__init__()
        
        self.bilstm = nn.LSTM(input_size=input_dim, hidden_size=32, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)     
        self.fc = nn.Linear(64, 1)

    def forward(self, state_input):
       
        x, _ = self.bilstm(state_input)
        x = self.dropout(x) 
        x = self.fc(x)
        
        return x

class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    """

    def __init__(
        self,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        *args,
        **kwargs,
        ):
        super(CustomNetwork, self).__init__()
       
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.actor = ActorNetwork(last_layer_dim_pi, last_layer_dim_vf)  # assuming input_dim=12, output_dim=3 for the actor network
        self.critic = CriticNetwork(last_layer_dim_pi)  # assuming input_dim=12 for the critic network

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        return self.actor(features), self.critic(features)
    
    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        
        return self.critic(features) 
    
    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        
        return self.critic(features)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False
        self.action_space = action_space
        self.observation_space = observation_space

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(INPUT_SEQUENCE,OUTPUT_SEQUENCE)