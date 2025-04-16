import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque
import yaml
import os
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional

class DQNNetwork(nn.Module):
    """
    Deep Q-Network for cloud-edge task scheduling.
    
    This network architecture includes dueling streams and noisy layers
    for better exploration and more stable training.
    """
    
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        
        # Feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Dueling architecture - separate value and advantage streams
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values