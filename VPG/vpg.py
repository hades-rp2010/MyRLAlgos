#Vanilla policy gradients in pytorch

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from helper import Policy_net, Value_net, discount_rewards

value_h = 128
policy_h = 128
value_lr = 1e-2
policy_lr = 5e-4
max_length = 200
epochs = 1000
episodes_per_epoch = 10


class VPG:
    def __init__(self, 
                 env,
                 device, 
                 policy_h = policy_h, 
                 value_h = value_h):
        self.env = env
        self.device = device
        self.policy_net = Policy_net(env.observation_space.shape[0],policy_h, env.action_space.n, device)
        self.value_net = Value_net(env.observation_space.shape[0], value_h, device)

    def train(self, 
              value_lr = value_lr, 
              policy_lr = policy_lr, 
              max_length = max_length, 
              epochs = epochs, 
              episodes_per_epoch = episodes_per_epoch):
        """
        Train both the value_net and the policy_net
        """

        value_optim = torch.optim.Adam(self.value_net.parameters(), lr = value_lr)
        policy_optim = torch.optim.Adam(self.policy_net.parameters(), lr = policy_lr)
        rewards = []
        for epoch in tqdm(range(epochs), desc="Training"):
            epoch_rewards = []
            for episode in range(episodes_per_epoch):
                observation, info = self.env.reset()
                episode_states = []
                episode_values = []
                episode_actions = []
                episode_action_dist = []
                episode_rewards = []
                done = False

                for i in range(max_length):
                    observation = torch.tensor(observation, dtype = torch.float32, device=self.device)
                    episode_states.append(observation)
                    value = self.value_net.forward(observation)
                    episode_values.append(value)
                    action_probs = self.policy_net.forward(observation)
                    action_distribution = torch.distributions.Categorical(action_probs)
                    action = action_distribution.sample()
                    episode_actions.append(action)
                    episode_action_dist.append(action_distribution)

                    observation, reward, done, _, _ = self.env.step(action.item())

                    episode_rewards.append(reward)

                    if done:
                        break


                discounted_rewards = discount_rewards(episode_rewards)

                #noramlize the discounted_rewards
                discounted_rewards = (discounted_rewards - np.mean(discounted_rewards))/(np.std(discounted_rewards) + 1e-5)

                #convert to a tensor
                discounted_rewards = torch.tensor(discounted_rewards, dtype = torch.float32, device=self.device, requires_grad=True)
                # print(f"discounted_rewards.shape: {discounted_rewards.shape}")

                episode_values = torch.tensor(episode_values, device=self.device, requires_grad=True).squeeze()     
                value_loss = nn.MSELoss()(episode_values, discounted_rewards)


                # NOTES TO SELF: I AM SURE THIS COULD HAVE BEEN DONE BETTER
                policy_loss = torch.tensor(0.0, device=self.device) 
                for j in range(len(episode_rewards) - 1):
                    #Using baseline as the value to reduce variance
                    advantage = discounted_rewards[j] - episode_values[j]

                    episode_policy_loss = - episode_action_dist[j].log_prob(episode_actions[j]) * advantage
                    policy_loss += episode_policy_loss
                epoch_rewards.append(np.sum(episode_rewards))

            #Fit the value function and the policy function
            value_optim.zero_grad()
            value_loss.backward()
            value_optim.step()
            policy_optim.zero_grad()
            policy_loss.backward()
            policy_optim.step()

            avg_episode_reward = np.mean(epoch_rewards[-episodes_per_epoch:])
            rewards.append(avg_episode_reward)

            if epoch % (epochs / 10) == 0:
                print("Epoch -> ", epoch + 1)
                print("Mean Reward", avg_episode_reward.item())
                print("Value loss -> ", value_loss.item())
                print("Policy loss -> ", policy_loss.item())

        self.env.close()

        plt.plot(rewards)
        plt.show()

    def save(self):
        pass

    def load(self):
        pass

    def eval(self, episodes):
        with torch.no_grad():
            """
            Evaluates the policy learnt for the specified number of episodes
            """
            imgs = []
            total_rewards = []
            for i in range(episodes):
                observation, _ = self.env.reset()
                episode_rewards = []
                done = False

                while not done:
                    observation = torch.tensor(observation, dtype = torch.float32)
                    action_probs = self.policy_net.forward(observation)
                    action_distribution = torch.distributions.Categorical(action_probs)
                    action = action_distribution.sample()

                    observation, reward, done, _, _ = self.env.step(action.item())
                    episode_rewards.append(reward)

                total = np.sum(episode_rewards)
                total_rewards.append(total)
                print("Total episode reward %f" % total)

            print("Average episode reward %f" % np.mean(total_rewards))
            self.env.close()


if __name__ == "__main__":
    import gymnasium as gym
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    #Hyperparameters

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VPG(env, device, policy_h, value_h)
    model.train()


    # JUST FOR INFERECE
    # CAREFUL RUNNING THIS ON A MAC CONNECTED TO A DISPLAY PORT, IT MAY CRASH THE SYSTEM
    new_env = gym.make("CartPole-v1", render_mode="human")
    eval_model = VPG(new_env, device, policy_h, value_h)
    eval_model.policy_net.load_state_dict(model.policy_net.state_dict())
    eval_model.value_net.load_state_dict(model.value_net.state_dict())
    eval_model.eval(10)
