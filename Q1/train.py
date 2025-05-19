import numpy as np
import gymnasium as gym
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple

device = 'cuda' if torch.cuda.is_available() else 'cpu'

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'log_prob', 'value'])

def make_env():
    # Create Pendulum-v1 environment
    env = gym.make("Pendulum-v1", render_mode='rgbarray')
    return env



class PPONet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 128), 
            nn.Tanh(),
            nn.Linear(128, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 128), 
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def get_act(self, state):
        # action distribution 
        mu = self.actor(state)
        std = torch.exp(self.log_std)
        distribution = torch.distributions.Normal(mu, std)

        # sample an action from distribution
        a = distribution.rsample()

        # log prob of the action
        log_prob = distribution.log_prob(a).sum(dim=-1)

        return a, log_prob
    
    def evaluate(self, state, act):
        value = self.critic(state).squeeze(-1)
        
        mu = self.actor(state)
        std = torch.exp(self.log_std)
        distribution = torch.distributions.Normal(mu, std)

        log_prob = distribution.log_prob(act).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        
        return value, log_prob, entropy



class PPOAgent:
    def __init__(self, obs_dim, act_dim, device):
        self.device = device
        self.model = PPONet(obs_dim, act_dim)
        self.model.to(self.device)
        
        self.ppo_epoches = 10
        self.buffer = []
        self.batch_size = 64

        self.gamma = 0.99
        ep = 0.2
        self.min_ratio = 1-ep
        self.max_ratio = 1+ep

        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.00025)

        # actor_lr = 3e-4
        # critic_lr = 1e-3
        # self.optimizer = optim.Adam([
        #     {'params': self.model.actor.parameters(), 'lr': actor_lr},
        #     {'params': self.model.critic.parameters(), 'lr': critic_lr}
        # ])


        self.c1 = 0.5
        self.c2 = 0.01

    def collect(self, state, next_state, action, reward, done, log_prob, value):
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done)
        log_prob = torch.tensor(log_prob, dtype=torch.float32)
        value = torch.tensor(value, dtype=torch.float32)

        self.buffer.append(Transition(state, action, reward, next_state, done, log_prob, value))

    def compute_gae(self, rews, values, dones, next_value, gamma=0.99, lam=0.95):
        rewards = []
        gae = 0
        values = list(values)
        values = values + [next_value]
        
        for t in reversed(range(len(rews))):
            delta = rews[t] + gamma * values[t+1] * (1 - dones[t].float()) - values[t]
            gae = delta + gamma * lam * (1 - dones[t].float()) * gae
            rewards.insert(0, gae + values[t])
        return rewards

    def get_experience(self):
        exp = Transition(*zip(*self.buffer))

        states = torch.stack(exp.state).to(self.device)
        actions = torch.stack(exp.action).to(self.device)
        old_log_probs = torch.stack(exp.log_prob).to(self.device)

        dones = torch.tensor(exp.done, dtype=torch.float32, device=self.device)
        values = torch.tensor(exp.value, dtype=torch.float32, device=self.device)

        rewards = exp.reward

        returns = torch.tensor(self.compute_gae(rews=rewards, values=exp.value, dones = exp.done, next_value=0), dtype=torch.float32).to(self.device)

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return states, actions, old_log_probs, advantages, returns

    def update(self):
        states, actions, old_log_probs, advantages, returns = self.get_experience()
        actor_losses = []
        critic_losses = []
        entropy_losses = []
        losses = []

        for _ in range(self.ppo_epoches):
            for i in range(0, len(states), self.batch_size):
                idx = range(i, min(i + self.batch_size, len(states)))

                # V_phi(s) , logP_theta(s, a), H_theta(s)
                value, log_prob, entropy = self.model.evaluate(states[idx], actions[idx])

                policy_ratio = torch.exp(log_prob - old_log_probs[idx])
                # print(policy_ratio)
                surr1 = policy_ratio * advantages[idx]
                surr2 = torch.clamp(policy_ratio, self.min_ratio, self.max_ratio) * advantages[idx]
                # print(surr1, surr2)
                clipped_surrogate_objective = torch.min(surr1, surr2).mean()

                actor_loss = -clipped_surrogate_objective

                critic_loss = F.mse_loss(value, returns[idx])

                loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy.mean()
                actor_losses.append(actor_loss.item())
                critic_losses.append(self.c1 * critic_loss.item())
                entropy_losses.append(-self.c2 * entropy.mean().item())
                losses.append(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # print(actor_losses.mean(), critic_losses.mean(), entropy_losses.mean())
        tqdm.write(f"Actor loss: {sum(actor_losses) / len(actor_losses)}, Critic loss: {sum(critic_losses) / len(critic_losses)}, Entropy loss: {sum(entropy_losses) / len(entropy_losses)}")
        return sum(losses) / len(losses)


def train(env, epoches):
    obs_dim = env.observation_space.shape[0] # 3
    act_dim = env.action_space.shape[0] # 1
    # act_limit = env.action_space.high[0] # 2
    
    agent = PPOAgent(obs_dim, act_dim, device)

    episode_rewards = deque(maxlen=10)
    max_steps = 2000

    for epoch in tqdm(range(epoches)):
        state, _  = env.reset()
        state = torch.tensor(state).to(device)

        agent.buffer = [] # empty the buffer
        ret = 0

        for step in range(max_steps):

            with torch.no_grad():
                action, log_prob = agent.model.get_act(state)
                value, _, _ = agent.model.evaluate(state, action)
                value = value.item()

            next_state, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy())
            done = terminated or truncated
            next_state = torch.tensor(next_state).to(device)
            ret += reward

            agent.collect(state, next_state, action, reward, done, log_prob, value)
            
            state = next_state

            if done:
                state, _ = env.reset()
                state = torch.tensor(state).to(device)
                episode_rewards.append(ret)
                ret = 0


        # episode_rewards.append(ret)
        loss = agent.update()
        torch.save(agent.model.state_dict(), "PPO.pth")

        # tqdm.write(f"Epoch {epoch}, Reward: {ret:.2f}, Loss: {loss:.4f}")
        avg_rew = np.mean(episode_rewards)
        tqdm.write(f"Epoch {epoch}, Average Reward: {avg_rew:.2f}, Loss: {loss:.4f}")


def main():
    env = make_env()
    epoches = 300
    train(env, epoches)

if __name__ == "__main__":
    main()