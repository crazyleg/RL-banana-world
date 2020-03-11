# Comparison of DQN, Dual DQN, Duelling DQN and prioritized expirience replay on Banana environment

## Overview
I've tested a number of deep RL approaches to solve enviroment proposed at Udacity deep RL nanodegree.
That's not a grid search or any other meaningfull comparison, due to a large number of hyperparamenters, that are not directly
comparable between methods. It is merely a set of random attempts, looking a qualitative difference between methods.

## Ideas tested
 - "Naive" DQN (N-DQN)
 - Dual DQN with soft updates (D-DQN-S)
 - Dual DQN with soft updates with Epsilon 0.98 (D-DQN-S-E)
 - Dual DQN with hard update (D-DQN-H)
 - Duelling DQN with soft updates (Duel-DQN)
 - Duelling DQN with soft updates and prioritized expirince replay (Duel-DQN - PER)
 
## Results

| Method        | Is solved? | Solved on episode | Max score | Max score on episode|
| ------------- |:----------:| -----------------:| ---------:|--------------------:|
| N-DQN         | no         |                   |  -0.09    |    290              |
| D-DQN-S       | yes        |       310         |   16.54   |    810              |
| D-DQN-S-E     | yes        |       580         |   16.41   |    990              |
| D-DQN-H       | yes        |       530         |   14.76   |    990              |
| Duel-DQN      | yes        |       340         |   17.01   |    770              |
| Duel-DQN - PER| yes        |       410         |   16.49   |    1000             |


