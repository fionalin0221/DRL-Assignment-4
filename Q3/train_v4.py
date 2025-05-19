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
    def __init__(self, obs_dim, act_dim, act_limit, batch_size = 256, gamma=0.99, alpha=1, tau=0.005, lr = 1e-4):
        self.actor = Actor(obs_dim, act_dim, act_limit)
        self.critic = Critic(obs_dim, act_dim)
        self.critic_target = Critic(obs_dim, act_dim)

        self.actor.load_state_dict(torch.load("SAC_actor_3.pth"))
        self.critic.load_state_dict(torch.load("SAC_critic_3.pth"))

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor.to(device)
        self.critic.to(device)
        self.critic_target.to(device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.replay_memory = deque([], maxlen=300000)

        self.batch_size = batch_size
        self.gamma = gamma
        # self.alpha = alpha
        self.tau = tau
        self.act_limit = act_limit

        self.reward_scale = 1.0

        self.log_alpha = torch.tensor(np.log(1)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -act_dim
        self.log_alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

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
            q_next = torch.min(q1_next, q2_next)- self.alpha * next_logp

            target_q = (self.reward_scale * reward + self.gamma * (1 - done) * q_next)

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Actor loss
        act_new, logp = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, act_new)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * logp - q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # train alpha
        alpha_loss = -(self.log_alpha *(logp + self.target_entropy).detach()).mean()
        self.log_alpha_optim.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optim.step()

        # Update target network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss, critic_loss, self.log_alpha


def train(env, epoches):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    print(obs_dim, act_dim, act_limit)

    agent = SACAgent(obs_dim, act_dim, act_limit)

    returns = []
    actor_losses = []
    critic_losses = []

    actor_loss, critic_loss = None, None
    max_step = 200

    progress_bar = tqdm(total=10000000, desc="Training Progress")

    for epoch in range(epoches):
        state, _ = env.reset()
        done = False
        step = 0
        ret = 0
        
        while not done:
        # for st in range(max_step):

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
                actor_loss, critic_loss, log_alpha = agent.update()
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
    
                # Update tqdm bar manually
                progress_bar.update(1)

            if done:
                break
        
        returns.append(ret)

        if actor_loss and critic_loss:
            tqdm.write(f"Episode {epoch} — Return: {ret:.2f}, Step: {step}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.6f}, log alpha: {log_alpha}")
        else:
            tqdm.write(f"Episode {epoch} — Return: {ret:.2f}, Step: {step}")

        if epoch % 10 == 0:
            torch.save(agent.actor.state_dict(), "SAC_actor_4.pth")
            torch.save(agent.critic.state_dict(), "SAC_critic_4.pth")

            save_plot(returns, actor_losses, critic_losses)

def save_plot(returns, actor_losses, critic_losses):
    episodes = np.arange(1, len(returns) + 1)
    steps = np.arange(1, len(actor_losses) + 1)

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(episodes, returns, label='Return', color='blue')
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(steps, actor_losses, label='Actor Loss', color='green')
    plt.xlabel("Steps")
    plt.ylabel("Actor Loss")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(steps, critic_losses, label='Critic Loss', color='red')
    plt.xlabel("Steps")
    plt.ylabel("Critic Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("sac_training_progress.png")
    plt.close()


def main():
    env = make_env()
    epoches = 20000
    train(env, epoches)

if __name__ == "__main__":
    main()