import torch 
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from src.utils.utils import layer_init



class ActorCritic(nn.Module):
    '''
    This is the actor critic model for the PPO/SAC algorithm.
    For SAC we don't need to make use of critic
    '''
    def __init__(self, envs):
        super().__init__()
        self.n_states = envs.single_observation_space.n
        self.n_actions = envs.single_action_space.n

        # Make a tabular policy
    
        self.actor = layer_init(nn.Linear(self.n_states, self.n_actions, bias=False))
        self.critic = layer_init(nn.Linear(self.n_states, 1, bias=False))
    
    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action = None, action_mask = None):
        logits = self.actor(x)
        if action_mask is not None:
            # make a highly negative number to have prob of zero for logits. 
            logits = logits - ( (1 - action_mask ) * 1e10) 
            
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def get_action(self, x, action = None, action_mask = None):
        logits = self.actor(x)
        if action_mask is not None:
            # make a highly negative number to have prob of zero for logits. 
            logits = logits - ( (1 - action_mask ) * 1e10) 
            
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        action_probs = probs.probs
        log_prob = F.log_softmax(logits, dim=-1)
        
        return action, log_prob, action_probs
    
    

    

class QNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.n_states = envs.single_observation_space.n
        self.n_actions = envs.single_action_space.n

        # Make a tabular policy
    
        self.network = layer_init(nn.Linear(self.n_states, self.n_actions, bias=False))

       

    def forward(self, x, action_masks = None):
        q_values =  self.network(x)
        # mask the non available actions
        if action_masks is not None:
            q_values = q_values - ( (1 - action_masks ) * 1e10)
        return q_values


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    if duration == 0:
        return start_e
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class DeepQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.n_states = envs.single_observation_space.n  # 716 states
        self.n_actions = envs.single_action_space.n      # 25 actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define the network architecture
        self.network = nn.Sequential(
            # First hidden layer: 716 -> 128
            nn.Linear(self.n_states, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            # Second hidden layer: 128 -> 64
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            # Output layer: 64 -> 25
            nn.Linear(64, self.n_actions)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x, action_masks=None):
        q_values = self.network(x)
        if action_masks is not None:
            q_values = q_values - ((1 - action_masks) * 1e10)
        return q_values