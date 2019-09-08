import gym
import numpy as np
import matplotlib.pyplot as plt

from gym.wrappers import Monitor
from collections import namedtuple, deque

gym.logger.set_level(40)

# Helper to create a experience tuple with named fields
makeExperience = namedtuple('Experience', 
                            field_names=['state', 
                                         'action', 
                                         'reward', 
                                         'next_state', 
                                         'done'])

def make_env(env_id, 
             use_monitor=False, 
             monitor_dir='recordings', 
             seed=42):
    """Instantiates the OpenAI Gym environment
    
    Args:
        env_id (string): OpenAI Gym environment ID
        use_monitor (bool): whether or not to use gym.wrappers.Monitor
        seed (int)
    """
    
    env = gym.make(env_id) # instantiate the environment
    
    if use_monitor: 
        env = Monitor(env, monitor_dir)
        
    env.seed(seed)
    
    return env
    

def run_env(env, get_action=None, max_t=1000, render=False):
    """Run actions against an environment.
    We pass a function in that could or not be wrapping an agent's actions
    
    Args:
        env (Environment)
        get_action (func): returns actions based on a state
        max_t (int): maximum number of timesteps
    """
    
    if get_action is None:
        get_action = lambda _: env.action_space.sample()
        
    state = env.reset()
    env.render()
    
    while True:
        action = get_action(state)
        state, reward, done, _ = env.step(action)
        env.render()
    
        if done: break
         
    env.close()


class EnvironmentAdapterForUnity():
    """Wrapper for Unity Environment.
    The idea is to have common interface for all the environments:
        OpenAI Gym envs, Unity envs, etc...
    
    Args:
        unity_env (UnityEnvironment)
        brain_name (str): 
            name of the brain responsible for deciding the actions of their 
            associated agents
    
    Attributes:
        unity_env (UnityEnvironment)
        brain_name (str)
        train_mode (bool)
    """
    
    def __init__(self, unity_env, brain_name):
        self.unity_env = unity_env
        self.brain_name = brain_name
        self.train_mode = True
        
        brain = unity_env.brains[brain_name]
        
        self.action_size = brain.vector_action_space_size
        self.state_size = brain.vector_observation_space_size
    
    def render():
        pass # Unity envs don't need to render
    
    def step(self, action):
        env_info = self.unity_env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = done = env_info.local_done[0]
        
        return next_state, reward, done, env_info
    
    def reset(self):
        env_info = self.unity_env.reset(train_mode=self.train_mode)[self.brain_name]
        return env_info.vector_observations[0]
    
    def close(self):
        self.unity_env.close()


def plot_scores(scores, 
                title='Deep Q-Network', 
                figsize=(15, 6), 
                polyfit_deg=None):
    """Plot scores over time. Optionally will draw a line showing the trend
    
    Args:
        scores (List of float)
        title (str)
        figsize (tuple of float)
            Default: (15, 6)
        polyfit_deg (int): degree of the fitting polynomial (optional)
    """
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(scores)
    
    max_score = max(scores)
    idx_max = np.argmax(scores)
    plt.scatter(idx_max, max_score, c='r', linewidth=3)
    
    if polyfit_deg is not None:
        x = list(range(len(scores)))
        degs = np.polyfit(x, scores, polyfit_deg)
        p = np.poly1d(degs)
        plt.plot(p(x), linewidth=3)
    
    plt.title(title)
    ax.set_ylabel('Score')
    ax.set_xlabel('Epochs')
    ax.legend(['Score', 'Trend', 'Max score: {}'.format(max_score)])