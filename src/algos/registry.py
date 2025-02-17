# src/algos/registry.py
from src.algos.ppo import run_ppo
from src.algos.dqn import run_dqn
from src.algos.sac import run_sac
from src.algos.qlearning import run_qlearning
from src.algos.sarsa import run_sarsa
from src.algos.deepdqn import run_dqn as run_deepdqn  # Add this import

def get_algo(algo):
    if algo == 'ppo':
        return run_ppo
    elif algo == 'dqn':
        return run_dqn
    elif algo == 'sac':
        return run_sac
    elif algo == 'qlearning':
        return run_qlearning
    elif algo == 'sarsa':
        return run_sarsa
    elif algo == 'deepdqn':  # Add this condition
        return run_deepdqn
    else:
        raise NotImplementedError("Unknown algo {}".format(algo))