# Log of experiments


### 9b8f5d8 - "Naive" solution - logs/experiment1
Idea: Based on a priori knowledge I took default solution from lunar lander and used it here, adjusting eps_decay to 0.98 in roder to see  very fast learing with potentially impaired max score.

problem solved - yes
episode solved - 310
max score - 16.54
max score episode - 810


dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.98):
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

default DQN
default Replay buffer

### 9b8f5d8 - "Naive" solution EPS 0.995 - logs/experiment2
Idea: Expect slower learning but higher score

problem solved - yes
episode solved - 580
max score - 16.41
max score episode - 990

### 9b8f5d8 - "Very Naive" solution - signle DQN - logs/experiment3
Idea: check how bad is single DQN

problem solved - no
max score - -0.09
max score episone - 290

### 5cf5a3a - "Very Naive" solution - dual DQN with hard step update - logs/experiment4
Idea: Check if classifacal DualDQN with hard udpate is better thatn proposed TAU soft update

problem solved - yes
episode solved - 530
max score - 14.76
max score episode - 990

### GIT - "Naive" solution - First best with duelling DQN - logs/experiment5
Idea: Check if duelling DQN improves over first solution 

problem solved - yes
episode solved - 340
max score - 17.01
max score episode - 770




