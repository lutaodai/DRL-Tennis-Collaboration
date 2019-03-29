from ddpg_agent import Agent
from multi_agents import MultiAgent
from config import Config


from unityagents import UnityEnvironment
import numpy as np
import random
import torch
from collections import deque
import matplotlib.pyplot as plt


env = UnityEnvironment(file_name="Tennis_Linux_NoVis/Tennis.x86_64")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=False)[brain_name]
config = Config(env_info, brain)


ma = MultiAgent(config)


def ddpg(n_episodes=3000, max_t=2000):
    all_scores = []
    scores_window = deque(maxlen=100)
    
    for i_episode in range(1, n_episodes+1):
        
        ma.reset()
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations           
        scores = np.zeros(config.env.n_agents)

        for i in range(max_t):
            actions = ma.act(states)
            env_info = env.step(actions)[brain_name]
            rewards = env_info.rewards
            next_states = env_info.vector_observations
            dones = env_info.local_done

            ma.step(states, actions, rewards, next_states, dones)

            scores += rewards
            states = next_states
            if any(dones):
                break
        
        max_score = np.max(scores)
        scores_window.append(max_score)
        all_scores.append(max_score)
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        for i, save_agent in enumerate(ma.ddpg_agents):
            torch.save(save_agent.actor_local.state_dict(), 'checkpoint_' + str(i) + '.actor.pth')
            torch.save(save_agent.critic_local.state_dict(), 'checkpoint_' + str(i) + '.critic.pth')
                
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-5, np.mean(scores_window)))
            for i, save_agent in enumerate(ma.ddpg_agents):
                torch.save(save_agent.actor_local.state_dict(), 'checkpoint_' + str(i) + '.actor.pth')
                torch.save(save_agent.critic_local.state_dict(), 'checkpoint_' + str(i) + '.critic.pth')
            break 
            
    return all_scores

scores = ddpg()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig("score2.png")