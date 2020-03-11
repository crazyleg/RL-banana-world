[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Solution to "Project 1: Navigation" of the Udacity "Deep Reinforcement Learning Nanodegree"

### Results
[Results of various approaches and report is here](https://github.com/crazyleg/RL-banana-world/blob/master/Report.md)

### Introduction

For this project, we train an agent to navigate (and collect bananas!) in a large, square world. The original setup of the exercise is to be found [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation)

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas. Each episode ends after a given number of steps.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Setup

1. A requirements.txt file containing the necessary packages for this project is included in this repository. Further details can also be found [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/).

2. The necessary banana environment to run the project can be downloaded from one of the links below for different operating systems:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    The environment needs to be called within the Jupyter notebook described below. The corresponding location is highlighted accordingly.
    
    (_For Windows users_) see also [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) concerning 32-bit version or 64-bit versions of the Windows operating system.

    (_For AWS_) see also: [enabling a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md) or use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

3. The environment file needs to be placed in the repository folder, and unzipped or decompressed.


### Solution

The notebook `Navigation-Solution.ipynb` contains the code to set up the environment and the outer episode iteration to solve the reinforcement problem. Our solution uses a (double) deep Q-learning network (only standard feedforward layers) and experience replay.

The agent, the deep Q-Network and memory buffer are implemented in the files `dqn_agent_duel_ddqn.py` and `model.py`. 

The winning solution is a mix of dual DQN with soft update, and Duelling DQN network architecture. To speed up training process Epsilon decay for desreased to 0.98.
