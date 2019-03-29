[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/15965062/47237461-d2a90b00-d3e7-11e8-96a0-f0c9a0b7ad1d.png "Algorithm"
[image2]: https://user-images.githubusercontent.com/15965062/47818424-a171f880-dd60-11e8-8885-c331b33597bd.png "Plot of Rewards"

# Report - Deep RL Project: Collaboration and Competition

### Implementation Details

The code for this project is ordered in 6 python files, and the demonstration training code and instructions in the notebook 'Tennis.ipynb'. 

1. 'model.py': Architecture and logic for the neural networks implementing the actor and critic for the chosen DDPG algorithm, as well as the DoubleAgent class which contains all agent's actors and crtics.

2. 'agent.py': Implements the DDPG and MADDPG agent classes, which include the logic for the stepping, acting, learning and the buffer to hold the experience data on which to train the agent, on the single and multi agent level, and uses 'model.py' to generate the local and target networks for the actor and critic. 

3. 'run.py': Main training loop or evalute loop logic.

4. 'env.py': Wrapper around the Unity environment.

5. 'stats.py': Statistics while training the agent, for printing and tensorboard.

6. 'main.py': Main function for running in command line.

7. 'Tennis.ipynb': Main training logic and usage instructions. Includes explainations about the environment, state and action space, goals and final results. The main training loop creates the agents and trains them using the MADDPG (details below) until satisfactory results. 

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
replay buffer size | int(1e6)
minibatch size | 256
discount factor | 0.99  
tau (soft update) | 1e-3
learning rate actor | 1e-3
learning rate critic | 1e-3
L2 weight decay | 0
UPDATE_EVERY | 4
NUM_UPDATES | 2
NOISE | 1.0
NOISE_DECAY | 1e-6
NOISE_SIGMA | 0.2

6. Network architecture:
    - Both the actor and critic are implemented using fully connected networks, with 2 hidden layers of 512 units each, batch normalization and Relu activation function, with Tanh activation at the last layer for the actors and one unit output for critics.
    - Input and output layers sizes are determined by the state and action space. Critics concatenate actions in 2nd fully-connected layer.
    - Training time until solving the environment takes around 17 minutes on AWS p2 instance with Tesla k80 GPU.
    - See 'model.py' for more details.

### Plot of results

As seen below, the environment is solved after 1392 episodes (average 100 episodes > 0.5), and achieves best average score of above 0.85 in 1467 episodes.
...
...
Episode: 1492   Avg: 0.512   BestAvg: 0.512   σ: 0.546  |  Tot. Steps: 59455   Secs: 1006      |  Buffer: 60947   NoiseW: 0.8781
...
...

Solved in 1392 episodes!

![Plot of Rewards][image2]

###  Ideas for future work

1. This DDPG implementation was very dependent on hyperparameters, noise settings and random seed. Solving the environment using PPO, TRPO or D4PG might allow a more robust solution to this task.

2. Solving the more challenging [SoccerTwos](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#soccer-twos) environment using edited versions of these same algorithms. This will also involve competitive as well as collaborative agents which lack in the Tennis environment.

Thanks Daniel Barbosa for statistics reference
