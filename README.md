# Navigation using Deep Q-Network

## Project Details
[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
The idea is to train an agent to navigate, and collect points — these are bananas!, who doesn't like them? 😋— in a large square world using Deep Q-Network.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a purple banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding the purple ones.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## Getting Started
TODO

## Instructions
TODO
