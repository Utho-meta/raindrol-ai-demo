import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
import pygame
import time
import os
from collections import deque
from env import RaindrolEnv
import matplotlib.pyplot as plt


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action).unsqueeze(1),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


def save_checkpoint(filepath, policy_net, optimizer, episode, steps_done, epsilon, episode_rewards):
    torch.save({
        'policy_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episode,
        'steps_done': steps_done,
        'epsilon': epsilon,
        'episode_rewards': episode_rewards,
    }, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, policy_net, optimizer):
    checkpoint = torch.load(filepath)
    policy_net.load_state_dict(checkpoint['policy_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    episode = checkpoint['episode']
    steps_done = checkpoint['steps_done']
    epsilon = checkpoint['epsilon']
    episode_rewards = checkpoint['episode_rewards']
    print(f"Loaded checkpoint from {filepath}, resuming from episode {episode}")
    return episode, steps_done, epsilon, episode_rewards


def train_dqn(env, episodes=500, batch_size=64, gamma=0.99,
              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500,
              target_update=10, buffer_capacity=10000, lr=1e-3,
              render_every=50, save_path="dqn_model.pth",
              checkpoint_path="checkpoint.pth", load_checkpoint_file=None):
    state_dim = env.reset().shape[0]
    action_dim = 18

    policy_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)

    start_episode = 0
    steps_done = 0
    epsilon = epsilon_start
    episode_rewards = []

    if load_checkpoint_file is not None and os.path.exists(load_checkpoint_file):
        start_episode, steps_done, epsilon, episode_rewards = load_checkpoint(
            load_checkpoint_file, policy_net, optimizer
        )
        target_net.load_state_dict(policy_net.state_dict())
        print(f"Resuming training from episode {start_episode}, epsilon={epsilon:.4f}, steps_done={steps_done}")
    else:
        if load_checkpoint_file is not None and os.path.exists(load_checkpoint_file):
            policy_net.load_state_dict(torch.load(load_checkpoint_file))
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Loaded model parameters from {load_checkpoint_file}, but training stats start from scratch.")
            steps_done = 0
            epsilon = epsilon_start
            start_episode = 0
            episode_rewards = []
        else:
            print("No checkpoint or model found, starting training from scratch.")

    for episode in range(start_episode, episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.FloatTensor(state).unsqueeze(0))
                    action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                current_q = policy_net(states).gather(1, actions)
                with torch.no_grad():
                    next_q = target_net(next_states).max(1, keepdim=True)[0]
                    target_q = rewards + gamma * next_q * (1 - dones)

                loss = F.mse_loss(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            steps_done += 1
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-steps_done / epsilon_decay)

        episode_rewards.append(total_reward)

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}, Steps: {env.steps_this_episode}, "
              f"Attacks: {env.stats_attacks}, Kills: {env.stats_kills}, "
              f"Combos: {env.stats_combos}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

        if episode % render_every == 0:
            torch.save(policy_net.state_dict(), save_path)
            save_checkpoint(checkpoint_path, policy_net, optimizer, episode, steps_done, epsilon, episode_rewards)
            test_env = RaindrolEnv(render_mode=True)
            test_state = test_env.reset()
            test_done = False
            test_reward = 0
            while not test_done:
                with torch.no_grad():
                    q_values = policy_net(torch.FloatTensor(test_state).unsqueeze(0))
                    test_action = q_values.argmax().item()
                test_state, r, test_done, _ = test_env.step(test_action)
                test_reward += r
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        test_done = True
                time.sleep(0.02)
            print(f"Test episode reward: {test_reward:.2f}")

    return policy_net, episode_rewards


if __name__ == "__main__":
    env = RaindrolEnv(render_mode=False)

    CHECKPOINT_FILE = "checkpoint.pth"
    # Move the checkpoint aside to start a fresh run.

    trained_net, rewards = train_dqn(env, episodes=2000, render_every=50,
                                     epsilon_decay=10000, epsilon_end=0.1,
                                     load_checkpoint_file=CHECKPOINT_FILE)

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.show()

    torch.save(trained_net.state_dict(), "dqn_final.pth")
    print("Training finished, model saved.")

    np.save("rewards.npy", rewards)
