import random
import torch
import torch.nn.functional as F
import numpy as np

class HPConfig:
    buffer_size = int(1e5)  
    batch_size = 128        
    gamma = 0.99            
    tau = 1e-3              
    lr_actor = 1e-4        
    lr_critic = 1e-4       
    weight_decay = 0        
    update_every = 1  
    num_updates = 4
    
class GeneralConfig:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 1
    
class NetworkConfig:
    def __init__(self, input_size, output_size, hidden_sizes=(256,128), lr=1e-3, output_activation=None):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.lr = lr
        self.output_activation = output_activation

class EnvConfig:
    def __init__(self, env_info, brain):
        states = env_info.vector_observations
        self.action_size = brain.vector_action_space_size
        
        self.state_size = states.shape[1]
        self.n_agents = len(env_info.agents)

class TrainConfig:
    n_episodes = 1000
    print_every = 100
    goal = 30.
    max_t = 1000
    
        
    
class Config:
    def __init__(self, env_info, brain):
        self.hp = HPConfig()
        self.general = GeneralConfig()
        self.env = EnvConfig(env_info, brain)
        self.actor = NetworkConfig(input_size=self.env.state_size,
                                   output_size=self.env.action_size,
                                   hidden_sizes=(256,128),
                                   lr=self.hp.lr_actor
        )
        self.critic = NetworkConfig(input_size=(self.env.state_size)*self.env.n_agents,
                                    output_size=1,
                                    hidden_sizes=(256,128),
                                    lr=self.hp.lr_critic
        )
        self.train = TrainConfig()
