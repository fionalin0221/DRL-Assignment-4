import gymnasium as gym
import numpy as np
import torch

from train_v6 import Actor, Critic

device = 'cpu'

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)

        self.actor = Actor(obs_dim=67, act_dim=21, act_limit=1.0).to(device)
        # self.critic = Critic(obs_dim=67, act_dim=21).to(device)
        
        self.actor.load_state_dict(torch.load("SAC_actor_10.pth"))
        self.actor.eval()
        # self.critic.load_state_dict(torch.load("SAC_critic_copy.pth"))

    def act(self, observation):
        state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
        action, _ = self.actor.sample(state)
                
        return action.detach().cpu().numpy()[0]

        # return self.action_space.sample()
