import torch
import torch.nn as nn
from src.utils.models import QNetwork, linear_schedule
from src.utils.utils import layer_init
from src.utils.utils import set_seeds, make_env, get_mask, encode_state

# Initialize the linear layer using layer_init
network = layer_init(nn.Linear(716, 25, bias=False))

# Load the state dictionary from the .pt file
state_dict = torch.load("models/dqn/dqn_seed_0_episode_499900.pt")

# Remove "network." prefix from the state dictionary keys
new_state_dict = {key.replace("network.", ""): value for key, value in state_dict.items()}

# Load the updated state dictionary into the model
network.load_state_dict(new_state_dict)

# Set the model to evaluation mode
network.eval()

# Verify the loaded model
print(network)



print(new_state_dict)
