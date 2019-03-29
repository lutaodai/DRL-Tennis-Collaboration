from ddpg_agent import Agent, ReplayBuffer

import numpy as np
import torch


class MultiAgent:
    """Meta agent that contains the two DDPG agents and shared replay buffer."""
    
    def __init__(self, config):
        self.config = config
        self.n_agents = config.env.n_agents
        self.ddpg_agents = [Agent(i, config) for i in range(self.config.env.n_agents)]
        # the shared replay buffer
        self.memory = ReplayBuffer(config)
        self.t_step = 0
    
    def reset(self):
        for agent in self.ddpg_agents:
            agent.reset()
        
    def step(self, states, actions, rewards, next_states, dones):
        states = states.reshape(1, -1)
        next_states = next_states.reshape(1, -1)
        self.memory.add(states, actions, rewards, next_states, dones)
        
        self.t_step = (self.t_step + 1) % self.config.hp.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.config.hp.batch_size:
                for _ in range(self.config.hp.num_updates):
                    # each agent does it's own sampling from the replay buffer
                    experiences = [self.memory.sample() for _ in range(self.config.env.n_agents)]
                    self.learn(experiences, self.config.hp.gamma)
                    
    def act(self, states, add_noise=True):
        # pass each agent's state from the environment and calculate it's action
        all_actions = []
        for agent, state in zip(self.ddpg_agents, states):
            action = agent.act(state , add_noise=True)
            all_actions.append(action)
        return np.array(all_actions).reshape(1, -1) # reshape 2x2 into 1x4 dim vector
    
    def learn(self, experiences, gamma):
        # each agent uses it's own actor to calculate next_actions
        all_next_actions = []
        for i, agent in enumerate(self.ddpg_agents):
            _, _, _, next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(self.config.general.device)
            next_state = next_states.reshape(-1, self.config.env.action_size, self.config.env.state_size) \
                        .index_select(1, agent_id).squeeze(1)
            next_action = agent.actor_target(next_state)
            all_next_actions.append(next_action)
            
        # each agent uses it's own actor to calculate actions
        all_actions = []
        for i, agent in enumerate(self.ddpg_agents):
            states, _, _, _, _ = experiences[i]
            agent_id = torch.tensor([i]).to(self.config.general.device)
            state = states.reshape(-1, self.config.env.action_size, self.config.env.state_size)\
                    .index_select(1, agent_id).squeeze(1)
            action = agent.actor_local(state)
            all_actions.append(action)
        
        # each agent learns from it's experience sample
        for i, agent in enumerate(self.ddpg_agents):
            agent.learn(i, experiences[i], gamma, all_next_actions, all_actions)
    
    
    
    
    
    
    