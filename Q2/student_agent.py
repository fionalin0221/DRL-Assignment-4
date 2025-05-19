import gymnasium
import numpy as np
import torch
from train import PPOAgent

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), np.float64)
        self.agent = PPOAgent(obs_dim=5, act_dim=1, device='cpu')
        self.agent.model.load_state_dict(torch.load('PPO.pth', map_location="cpu"))
        self.agent.model.eval()

    def act(self, observation):
        state = torch.tensor(observation, dtype=torch.float32)
        action, _ = self.agent.model.get_act(state)
        return action.detach().cpu().numpy()
    
        # return self.action_space.sample()
