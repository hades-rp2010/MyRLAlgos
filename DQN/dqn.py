import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import LinearQNetwork, CNNQNetwork, FramePreprocessor  

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
LEARNING_RATE = 3e-4
EPSILON = 0.1
GAMMA = 0.99
MAX_BUFFER_LENGTH = 10000
MAX_TRAJ_LENGTH = 10000
BATCH_SIZE = 256


class DQNAgent:
    def __init__(self, env, q):
        self.env = env
        self.q = q.to(device) # THe Q network: a neural network that takes in the state and outputs the Q values for each action
        self.device = device

    def select_action(self, obs, eps = EPSILON):
        if torch.rand(1).item() < eps:
            action = torch.tensor(self.env.action_space.sample(), dtype = torch.float32, device=self.device)
        else:
            obs = obs.to(self.device)
            q_values = self.q(obs)
            action = torch.argmax(q_values)

        return action
    
    def train(self, episodes, gamma = GAMMA, eps = EPSILON, max_buffer_length = MAX_BUFFER_LENGTH, max_traj_length = MAX_TRAJ_LENGTH, batch_size = BATCH_SIZE):

        epoch_rewards = []
        episode_durations = []
        optimizer = optim.Adam(self.q.parameters(), lr = LEARNING_RATE)
        replay_buffer = deque(maxlen = max_buffer_length)
        for episode in tqdm(range(episodes)):
            observation, _ = self.env.reset()
            episode_rewards = []
            done = False
                
            for i in range(max_traj_length):
                # Play the game and collect experience
                observation = torch.tensor(observation, dtype = torch.float32, device=self.device)
                action = self.select_action(observation, eps = eps)
                next_observation, reward, done, _, _ = self.env.step(int(action.item()))
                episode_rewards.append(reward)
                next_observation = torch.tensor(next_observation, dtype = torch.float32, device=self.device)
                reward = torch.tensor(reward, dtype = torch.float32, device=self.device)
                done = torch.tensor(done, dtype = torch.float32, device=self.device)
                transition = (observation, action, reward, done, next_observation)
                replay_buffer.append(transition)
                observation = next_observation
                if done:
                    episode_durations.append(i)
                    break
                
                # train your network
                if len(replay_buffer) >= batch_size:
                    # Sample a batch of transitions from the replay buffer
                    batch = random.sample(replay_buffer, batch_size)
                    observations, actions, rewards, dones, next_observations = zip(*batch)
                    observations = torch.stack(observations).to(self.device)
                    next_observations = torch.stack(next_observations).to(self.device)
                    rewards = torch.stack(rewards).to(self.device)
                    dones = torch.stack(dones).to(self.device)
                    actions = torch.stack(actions).long().to(self.device)
                    with torch.no_grad():
                        target_q_values = (rewards + gamma * (torch.ones_like(dones) - dones) * self.q(next_observations).max(dim=1)[0]).unsqueeze(1)
                    q_values = self.q(observations).gather(1, actions.view(-1, 1))
                    loss = nn.MSELoss()(q_values, target_q_values)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            epoch_rewards.append(np.sum(episode_rewards))

        self.env.close()

        # Plot the rewards 
        plt.plot(epoch_rewards)
        plt.show()

    def eval(self, episodes):
        rewards = []
        for episode in range(episodes):
            observation, _ = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                self.env.render()
                observation = torch.tensor(observation, dtype = torch.float32, device=self.device)
                action = self.select_action(observation, eps = 0.0) # No exploration during evaluation
                next_observation, reward, done, _, _ = self.env.step(int(action.item()))
                episode_reward += reward
                observation = next_observation
            rewards.append(episode_reward)

        plt.plot(rewards)
        plt.show()
    


if __name__ == "__main__":
    import gymnasium as gym
    import ale_py

    gym.register_envs(ale_py)

    game ="CartPole-v1"
    env = gym.make(game, render_mode="rgb_array")
    env.reset()

    # DQN
    q_network = LinearQNetwork(env.observation_space.shape[0], env.action_space.n)
    agent = DQNAgent(env, q_network)
    agent.train(episodes = 300)

    # evaluation:
    eval_env = gym.make(game, render_mode="human")
    eval_agent = DQNAgent(eval_env, q_network)
    eval_rewards = eval_agent.eval(episodes = 10)
    