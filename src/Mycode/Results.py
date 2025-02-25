import sys
import os
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import random
import icu_sepsis



# Utility functions
def calculate_discounted_return(reward_list, gamma = 0.99):
    discounted_return = 0
    for reward in reversed(reward_list):
        discounted_return = reward + gamma * discounted_return
    return discounted_return

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

def encode_state(obs, n_states):
    obs = torch.Tensor(obs)
    return nn.functional.one_hot(obs.long(), n_states).float()

def get_mask(info, num_envs=1, n_actions=25):
    allowed_actions = info['admissible_actions']
    mask = np.zeros(n_actions)
    for action in allowed_actions:
        mask[action] = 1
    return torch.Tensor(mask).unsqueeze(0)

def layer_init(layer):
    torch.nn.init.constant_(layer.weight, 0.0)
    return layer

# Define the network architecture
class DeepQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        
        self.network = nn.Sequential(
            layer_init(nn.Linear(self.n_states, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, self.n_actions))
        )

    def forward(self, x, action_masks=None):
        q_values = self.network(x)
        if action_masks is not None:
            q_values = q_values - ((1 - action_masks) * 1e10)
        return q_values

def evaluate_saved_deepqn(model_path, start_seed=0, end_seed=999, num_episodes_per_seed=100):
    """
    Evaluate a saved Deep Q-Network model using the same seeds as process_data.py (0-999).
    """
    np.set_printoptions(precision=None, suppress=True, floatmode='fixed')
    
    all_returns = []
    all_episode_lengths = []
    all_discounted_returns = []

    env = gym.make('Sepsis/ICU-Sepsis-v2')
    q_network = DeepQNetwork(env)
    state_dict = torch.load(model_path, weights_only=True)
    q_network.load_state_dict(state_dict)
    q_network.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = q_network.to(device)

    # Use exact same seed range as process_data.py
    for seed in range(start_seed, end_seed + 1):
        set_seeds(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        
        seed_returns = []
        seed_lengths = []
        seed_discounted_returns = []

        for episode in range(num_episodes_per_seed):
            state, info = env.reset(seed=seed)
            episode_reward = 0
            steps = 0
            done = False
            episode_rewards = []
            
            while not done:
                state_encoded = encode_state(np.array([state]), env.observation_space.n)
                action_mask = get_mask(info)
                
                state_encoded = state_encoded.to(device)
                action_mask = action_mask.to(device)
                
                with torch.no_grad():
                    q_values = q_network(state_encoded, action_mask)
                    action = torch.argmax(q_values, dim=1).cpu().item()
                
                next_state, reward, terminated, truncated, info = env.step(action)
                
                episode_rewards.append(reward)
                episode_reward += reward
                steps += 1
                state = next_state
                done = terminated or truncated

            discounted_return = calculate_discounted_return(episode_rewards)
            
            seed_returns.append(episode_reward)
            seed_lengths.append(steps)
            seed_discounted_returns.append(discounted_return)

        seed_mean_return = np.mean(seed_returns)
        all_returns.append(seed_mean_return)
        all_episode_lengths.append(np.mean(seed_lengths))
        all_discounted_returns.append(np.mean(seed_discounted_returns))
        
        if seed % 10 == 0:  # Print progress every 10 seeds
            print(f"Seed {seed} completed. Mean return: {seed_mean_return}")

    statistics = {
        'mean_return': np.mean(all_returns),
        'std_return': np.std(all_returns),
        'mean_discounted_return': np.mean(all_discounted_returns),
        'std_discounted_return': np.std(all_discounted_returns),
        'mean_episode_length': np.mean(all_episode_lengths),
        'std_episode_length': np.std(all_episode_lengths),
        'min_return': np.min(all_returns),
        'max_return': np.max(all_returns),
        'all_returns': np.array(all_returns),
        'all_discounted_returns': np.array(all_discounted_returns),
        'all_episode_lengths': np.array(all_episode_lengths),
        '95_confidence_interval': (
            np.mean(all_returns) - 1.96 * np.std(all_returns) / np.sqrt(len(all_returns)),
            np.mean(all_returns) + 1.96 * np.std(all_returns) / np.sqrt(len(all_returns))
        )
    }

    print("\nFinal Evaluation Results:")
    for key, value in statistics.items():
        if key not in ['all_returns', 'all_discounted_returns', 'all_episode_lengths']:
            if isinstance(value, tuple):
                print(f"{key}: ({value[0]}, {value[1]})")
            else:
                print(f"{key}: {value}")

    return statistics

if __name__ == "__main__":
    #model_path = r"C:\Users\Jordan Lankford\Documents\GitHub\FineTune-DQN\models\dqn\dqn_final_seed_0.pt"
    model_path = r"C:\Users\Jordan Lankford\Documents\GitHub\FineTune-DQN\models\dqn\standarddqn_final_seed_0.pt"

    stats = evaluate_saved_deepqn(
        model_path, 
        start_seed=0,
        end_seed=999,
        num_episodes_per_seed=100
    )