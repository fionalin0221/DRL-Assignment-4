import gymnasium as gym
import numpy as np
import torch
from train import PPOAgent

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)
        self.agent = PPOAgent(obs_dim=3, act_dim=1, device='cpu')
        self.agent.model.load_state_dict(torch.load('PPO.pth', map_location="cpu"))
        self.agent.model.eval()

    def act(self, observation):
        state = torch.tensor(observation)
        action, _ = self.agent.model.get_act(state)
        return action.detach().cpu().numpy()
        
        # return self.action_space.sample()