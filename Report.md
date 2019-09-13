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
_Experience Replay_ is a technique we use to decorrelate transitions observed by the agent and that are stored for later resused and learnt from. It has been shown that this greatly stabilizes and improves the DQN training procedure. These transitions are stored in a buffer, **buffer_size=1e5** , and will be sampled in batches, **batch_size=64**, 

The _discount factor_ (γ) is a measure of how far ahead in time the algorithm looks. If we wanted to prioritise rewards in the distant future, we'd keep the value closer to one. On the other hand if we wanted to consider only rewards in the immediate future, then we'd use a discount factor closer to zero. I'll be using **gamma=0.99**

We use _Fixed Q-targets_ to prevent the "chasing a moving target" effect when using the same parameters (weights) for estimating the target and the Q value. This is because there is a big correlation between the TD target and the parameters we are changing. The solution is using a second network whose weights will be [softly updated](https://github.com/jscriptcoder/Navigation-Deep-Q-Network/blob/master/agent/agent.py#L228), with interpolation parameter **tau=1e-3**, and that we'll be using to calculate the TD target.

We'll be using Adam optimizer with a learning rate **lr=5e-4** for our models. 

How ofter we're gonna learn is dictated by the parameter **update_every=4**, which means every 4 steps we're gonna learn from previous experiences and update the weights of our networks.

## Plot of Rewards
```
A plot of rewards per episode is included to illustrate that the agent is able to receive an average reward (over 100 episodes) of at least +13. The submission reports the number of episodes needed to solve the environment.
```

## Ideas for Future Work
```
The submission has concrete future ideas for improving the agent's performance.
```
