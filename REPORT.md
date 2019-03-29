[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/15965062/47237461-d2a90b00-d3e7-11e8-96a0-f0c9a0b7ad1d.png "Algorithm"
[image2]: https://raw.githubusercontent.com/lutaodai/DRL-Tennis-Collaboration/master/score.png "Plot of Rewards"

# Report - Deep RL Project: Collaboration and Competition

This report is organized based on [Akhiad Bercovich](https://github.com/akhiadber/DeepRL-Tennis-Collab/blob/master/REPORT.md)'s report.

### Implementation Details

1. `config.py`: Configuration files for training the model;
1. `model.py`: Actor and Critc Network classes;
1. `ddpg_agent.py`: Agent, ReplayBuffer and OUNoise classes; The Agent class makes use of the Actor and Critic classes from `model.py`, the ReplayBuffer class and the OUNoise class;
1. `multi_agents.py`: MultiAgent class defining multiple agents based on the `Agent` class;
1. `run.py`: Script which will train the agent. Can be run directly from the terminal;
1. `checkpoint_[01].actor.pth`: Contains the weights of successful Actor Networks;
1. `checkpoint_[01].critic.pth`: Contains the weights of successful Critic Networks.

To train the model, simply adjust parameters in the `config.py` file, and then run
```bash
python run.py
```

### Learning Algorithm

The agent is trained using the MADDPG algorithm. This is an extension of DDPG agent to multi-agent situations (more details below). 

References:
1. [MADDPG paper](https://arxiv.org/abs/1706.02275)

2. [DDPG Paper](https://arxiv.org/pdf/1509.02971.pdf)

3. [DDPG-pendulum implementation](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum)

4. DDPG Algorithm details: 

![Algorithm][image1]


4. Short explanation (refer to the papers for further details):
    - Q-Learning is not straighforwardly applied to continuous tasks due to the argmax operation over infinite actions in the continuous domain. DDPG can be viewed as an extension of Q-learning to continuous tasks.

    - DDPG was introduced as an actor-critic algorithm, although the roles of the actor and critic here are a bit different then the classic actor-critic algorithms. Here, the actor implements a current policy to deterministically map states to a specific "best" action. The critic implemets the Q function, and is trained using the same paradigm as in Q-learning, with the next action in the Bellman equation given from the actor's output. The actor is trained by the gradient from maximizing the estimated Q-value from the critic, when the actor's best predicted action is used as input to the critic.
    
    - As in Deep Q-learning, DDPG also implements a replay buffer to gather experiences from the agent (or the multiple parallel agents in the 2nd version of the stated environment). 
    
    - In order to encourage exploration during training, Ornstein-Uhlenbeck noise is added to the actors selected actions. I also needed to decay this noise using an epsilon hyperparameter to achieve best results.
    
    - Another fine detail is the use of soft updates (parameterized by tau below) to the target networks instead of hard updates as in the original DQN paper.
    
    - In order to use DDPG in multi-agent environments, each agent (in this case 2) has its own critic and actor networks, the experience from both is sent to a shared buffer, the actors are trained on states sampled from the buffer from one agent, and critics on a concatenation of all agents' sampled actions and states. This flexible framework allows training in collaborative, competitive or mixed multi-agent environments.
    
6. Hyperparameters:

Parameter | Value
--- | ---
replay buffer size | int(1e5)
minibatch size | 128
discount factor | 0.99  
tau (soft update) | 1e-3
learning rate actor | 1e-4
learning rate critic | 1e-4
L2 weight decay | 0
update frequency (episode) | 1
numer of updates (per episode) | 4

6. Network architecture:
    - Both the actor and critic are implemented using fully connected networks, with 2 hidden layers of 512 units each, batch normalization and Relu activation function, with Tanh activation at the last layer for the actors and one unit output for critics.
    - Input and output layers sizes are determined by the state and action space. Critics concatenate actions in 2nd fully-connected layer.
    - Training time until solving the environment takes around 17 minutes on AWS p2 instance with Tesla k80 GPU.
    - See 'model.py' for more details.
    
### Training screen output
```
Episode 100     Average Score: 0.02
Episode 200     Average Score: 0.01
Episode 300     Average Score: 0.04
Episode 400     Average Score: 0.06
Episode 500     Average Score: 0.10
Episode 600     Average Score: 0.13
Episode 700     Average Score: 0.17
Episode 800     Average Score: 0.23
Episode 900     Average Score: 0.28
Episode 1000    Average Score: 0.31
Episode 1100    Average Score: 0.39
Episode 1175    Average Score: 0.50
Environment solved in 1170 episodes!    Average Score: 0.50  
```

### Plot of results

As seen below, the environment is solved after 1170 episodes (average 100 episodes > 0.5)

![Plot of Rewards][image2]

###  Ideas for future work

1. This DDPG implementation was very dependent on hyperparameters, noise settings and random seed. Solving the environment using PPO, TRPO or D4PG might allow a more robust solution to this task.

2. Solving the more challenging [SoccerTwos](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#soccer-twos) environment using edited versions of these same algorithms. This will also involve competitive as well as collaborative agents which lack in the Tennis environment.
