import numpy as np
import gymnasium as gym
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env

device = 'cuda' if torch.cuda.is_available() else 'cpu'

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

def make_env():
    # Create environment with image observations
    env_name = "humanoid-walk"
    env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    return env


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)
        self.act_limit = act_limit

    def forward(self, state):
        x = self.fc(state)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self(state)
        distribution = torch.distributions.Normal(mean, std)
        z = distribution.rsample()
        action = torch.tanh(z)
        log_prob = distribution.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action * self.act_limit, log_prob.sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, act):
        xu = torch.cat([obs, act], dim=-1)
        return self.q1(xu), self.q2(xu)

class SACAgent:
    def __init__(self, obs_dim, act_dim, act_limit, batch_size = 256, gamma=0.99, alpha=10, tau=0.005, lr = 3e-4):
        self.actor = Actor(obs_dim, act_dim, act_limit)
        self.critic = Critic(obs_dim, act_dim)
        self.critic_target = Critic(obs_dim, act_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor.to(device)
        self.critic.to(device)
        self.critic_target.to(device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.replay_memory = deque([], maxlen=100000)

        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.act_limit = act_limit

        self.reward_scale = 1

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        act, _ = self.actor.sample(state)
        return act.detach().cpu().numpy()[0]
    
    def cache(self, state, action, reward, next_state, done):
        # save experience into replay memory
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        self.replay_memory.append(Transition(state, action, reward, next_state, done))
    
    def sample(self):
        random_samples = random.sample(self.replay_memory, self.batch_size) # list of Transition
        batch = Transition(*zip(*random_samples)) # Transition of list

        state_batch = torch.stack(batch.state).to(device)
        next_state_batch = torch.stack(batch.next_state).to(device)

        action_batch = torch.stack(batch.action).to(device)
        reward_batch = torch.stack(batch.reward).unsqueeze(1).to(device)
        done_batch = torch.stack(batch.done).unsqueeze(1).to(device)

        return state_batch, next_state_batch, action_batch, reward_batch, done_batch

    def update(self):
        torch.autograd.set_detect_anomaly(True)
        
        state, next_state, action, reward, done = self.sample()
        
        # Critic loss
        with torch.no_grad():
            next_act, next_logp = self.actor.sample(next_state)
            q1_next, q2_next = self.critic_target(next_state, next_act)
            q_next = torch.min(q1_next, q2_next)
            
            target_q = (self.reward_scale * reward + self.gamma * (1 - done) * (q_next))

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # Actor loss
        act_new, logp = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, act_new)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * logp - q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Update target network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss, critic_loss


def train(env, epoches):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    print(obs_dim, act_dim, act_limit)

    agent = SACAgent(obs_dim, act_dim, act_limit)

    actor_loss, critic_loss = None, None

    for epoch in range(epoches):
        state, _ = env.reset()
        done = False
        step = 0

        ret = 0
        while not done:

            action = agent.select_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ret += reward

            agent.cache(state, action, reward, next_state, done)

            # try:
            #     # Optionally render the environment here
            #     frame = env.render()  # This will return the rendered image (RGB array)
                
            #     # Display the frame using matplotlib
            #     plt.imshow(frame)
            #     plt.axis('off')  # Turn off axis for clean display
            #     plt.show(block=False)
            #     plt.pause(0.1)  # Pause briefly to allow for rendering
            #     plt.clf()  # Clear the figure between frames

            # except KeyboardInterrupt:
            #     print("\nTraining interrupted by user.")

            state = next_state
            step += 1

            if len(agent.replay_memory) >= agent.batch_size:
                actor_loss, critic_loss = agent.update()

            if done:
                break
        
        # if actor_loss and critic_loss:
        print(f"Episode {epoch} — Return: {ret:.2f}, Step: {step}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

        if epoch % 10 == 0:
            torch.save(agent.actor.state_dict(), "SAC_actor.pth")
            torch.save(agent.critic.state_dict(), "SAC_critic.pth")

def main():
    env = make_env()
    epoches = 20000
    train(env, epoches)

if __name__ == "__main__":
    main()