# parallel_fitness.py
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from multiprocessing import Pool
import time
import icu_sepsis

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        
        self.network = nn.Sequential(
            nn.Linear(self.n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_actions)
        )

    def forward(self, x, action_masks=None):
        q_values = self.network(x)
        if action_masks is not None:
            q_values = q_values - ((1 - action_masks) * 1e10)
        return q_values

def evaluate_chunk(args):
    """
    Evaluate a large chunk of episodes in one process
    """
    chunk_size, model_state_dict = args
    
    # Create environment and model once per chunk
    env = gym.make('Sepsis/ICU-Sepsis-v2')
    model = QNetwork(env)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    total_reward = 0
    
    for _ in range(chunk_size):
        state, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                state_encoded = torch.Tensor(np.eye(env.observation_space.n)[state]).unsqueeze(0)
                action_mask = torch.Tensor(np.zeros(env.action_space.n))
                action_mask[info['admissible_actions']] = 1
                action_mask = action_mask.unsqueeze(0)
                
                q_values = model(state_encoded, action_mask)
                action = torch.argmax(q_values).item()
            
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            state = next_state
            done = terminated or truncated
        
        total_reward += episode_reward
    
    env.close()
    return total_reward

def parallel_fitness(model_state_dict, num_processes=4, num_episodes=10000):
    """
    Parallel fitness evaluation with larger chunks
    """
    chunk_size = num_episodes // num_processes
    args = [(chunk_size, model_state_dict) for _ in range(num_processes)]
    
    start_time = time.perf_counter()
    with Pool(num_processes) as pool:
        chunk_rewards = pool.map(evaluate_chunk, args)
    end_time = time.perf_counter()
    
    total_episodes = chunk_size * num_processes
    fitness = sum(chunk_rewards) / total_episodes
    print(f"Evaluated {total_episodes} episodes in {end_time - start_time:.2f} seconds")
    return -fitness

if __name__ == '__main__':
    env = gym.make('Sepsis/ICU-Sepsis-v2')
    model = QNetwork(env)
    state_dict = model.state_dict()
    result = parallel_fitness(state_dict, num_processes=4, num_episodes=10000)
    print(f"Test result: {result}")