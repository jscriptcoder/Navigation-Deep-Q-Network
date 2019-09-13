# Report: Navigation using Deep Q-Network

## Learning Algorithm
```
The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks.
```
What we're dealing with here is an envirornment with continuous state space and discrete action space with 4 possible actions. Deep Q-Network is an excellent choise to address this problem.

I'm gonna solve this environment using [vanilla Deep Q-Netwok](http://www.readcube.com/articles/10.1038/nature14236) (with fixed Q-targets) and then we'll try different improvements such as:

- Deep Reinforcement Learning with Double Q-learning. [Paper](https://arxiv.org/abs/1509.06461)
- Dueling Network Architectures for Deep Reinforcement Learning. [Paper](https://arxiv.org/abs/1511.06581)
- Prioritized Experience Replay. [Paper](https://arxiv.org/abs/1511.05952)
- Noisy Networks for Exploration. [Paper](https://arxiv.org/abs/1706.10295)

To finish off, I'll use all of them at once and hopefully we'll see a big performance boost.

### Hyperparameters
_Experience Replay_ is a technique we use to decorrelate transitions observed by the agent and that are stored for later resused and learnt from. It has been shown that this greatly stabilizes and improves the DQN training procedure. These transitions are stored in a buffer, ```buffer_size=1e5``` , and will be sampled in batches, ```batch_size=64```, 

The _discount factor_ is a measure of how far ahead in time the algorithm looks. If we wanted to prioritise rewards in the distant future, we'd keep the value closer to one. On the other hand if we wanted to consider only rewards in the immediate future, then we'd use a discount factor closer to zero. I'll be using ```gamma=0.99```

We use _Fixed Q-targets_ to prevent the "chasing a moving target" effect when using the same parameters (weights) for estimating the target and the Q value. This is because there is a big correlation between the TD target and the parameters we are changing. The solution is using a second network whose weights will be [softly updated](https://github.com/jscriptcoder/Navigation-Deep-Q-Network/blob/master/agent/agent.py#L228), with interpolation parameter ```tau=1e-3```, and that we'll be using to calculate the TD target.

We'll be using Adam optimizer with a learning rate ```lr=5e-4``` for our models. 

How ofter we're gonna learn is dictated by the parameter ```update_every=4```, which means every 4 steps we're gonna learn from previous experiences and update the weights of our networks.

### Solutions
1. **Vanilla DQN**:
Deep Q-learning algorithm using a simple Experience Replay buffer, Fixed Q-Target improvement and epsilon-greedy exploration method. This is an [excellent article](https://medium.com/@jonathan_hui/rl-dqn-deep-q-network-e207751f7ae4) about this algorithm.

2. **Double Q-Network**:
Double DQNs, or double Learning, was introduced by [Hado van Hasselt](https://papers.nips.cc/paper/3964-double-q-learning). This method handles the problem of the overestimation of Q-values. There is a problem when calculating the TD target: how are we sure that the best action for the next state is the action with the highest Q-value?. As a consequence, at the beginning of the training we don't have enough information about the best action to take. Therefore, taking the maximum Q-value as the best action can lead to false positives (overestimation). There is an [interesting article](https://towardsdatascience.com/double-deep-q-networks-905dd8325412) explaining this solution.

3. **Dueling Q-Network**:
The Q-values correspond to how good it is to be at that state and taking an action at that state ```Q(s,a)```. The idea behind this architecture is to separate the estimator of two elements: ```V(s)```, the value of being at that state and ```A(s,a)```, the advantage of taking that action at that state (how much better is to take this action versus all other possible actions at that state), using two new streams, and then we combine them through a special aggregation layer to get an estimate of ```Q(s,a)```. [Here](https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df#af34) there is a better explanation of the architecture.

4. **DQN with Prioritized Experience Replay**:
Prioritized Experience Replay (PER) was introduced in 2015 by [Tom Schaul](https://arxiv.org/search/?searchtype=author&query=Schaul%2C+T). In the famous DQN algorithm, a buffer of past experiences is used to stabilize training by decorrelating the training examples in each batch used to update the neural network. But some experiences may be more important than others for our training, and might occur less frequently. Because we sample the batch uniformly (selecting the experiences randomly) these rich experiences that occur rarely have practically no chance to be selected. with PER we try to change the sampling distribution by using a criterion to define the priority of each tuple of experience. You can find more details about experience replay and PER in [this article](https://medium.com/@badamivikas/deeper-intuition-on-prioritized-experience-replay-33fcc55de44e).

5. **DQN using Noisy Networks for exploration**:
One fundamental problem of Reinforcement Learning is Exploration/Exploitation dilemma. This issue raises from the fact that our agent needs to keep the balance between exploring the environment and using what's learned from this exploration. There are different ways to address this problem. Most common one is using epsilon-greedy: with some probability epsilon (given as hyperparameter) your agent takes a random step instead of acting according to policy it has learned. DeepMind proposes very simple and surprisingly efficient way of tackling the above issue. The method basically consists of adding the gaussian noise to the last (fully-connected) layers of the network. You can read more about the technique in this [blog post](https://medium.com/@shmuma/summary-noisy-networks-for-exploration-c8ba6e2759c7).

6. **All of them combined**:
We'll see in a moment what it is to use all this improvements at the same time.

## Plot of Rewards
```
A plot of rewards per episode is included to illustrate that the agent is able to receive an average reward (over 100 episodes) of at least +13. The submission reports the number of episodes needed to solve the environment.
```

## Ideas for Future Work
```
The submission has concrete future ideas for improving the agent's performance.
```
