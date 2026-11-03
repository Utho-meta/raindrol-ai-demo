import numpy as np

rewards = np.load("rewards.npy")

print(rewards)

print(f"Episodes: {len(rewards)}")
print(f"Min: {rewards.min():.2f}")
print(f"Max: {rewards.max():.2f}")
print(f"Mean: {rewards.mean():.2f}")
print(f"Last 10: {rewards[-10:]}")
