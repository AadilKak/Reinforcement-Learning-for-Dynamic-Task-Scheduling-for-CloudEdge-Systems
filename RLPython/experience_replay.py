import numpy as np
from collections import namedtuple

# Define named tuple for storing experiences
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class EnhancedReplayMemory:
    """Enhanced replay buffer with prioritized experience replay"""
    
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=100000):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha  # Priority exponent
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.beta = beta_start
        self.frame = 1
        self.eps = 1e-6  # Small constant to ensure all priorities > 0
        
    def update_beta(self):
        """Update beta parameter for importance sampling"""
        progress = min(self.frame / self.beta_frames, 1.0)
        self.beta = self.beta_start + progress * (self.beta_end - self.beta_start)
        self.frame += 1
        
    def push(self, *args):
        """Save a transition with maximum priority"""
        max_priority = self.priorities.max() if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(Experience(*args))
        else:
            self.memory[self.position] = Experience(*args)
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Sample a batch of experiences with prioritization"""
        if len(self.memory) == 0:
            return [], [], []
            
        if len(self.memory) < self.capacity:
            probs = self.priorities[:len(self.memory)]
        else:
            probs = self.priorities
            
        # Calculate sampling probabilities
        probs = probs ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.memory), batch_size, p=probs[:len(self.memory)])
        
        # Calculate importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Get the sampled experiences
        experiences = [self.memory[idx] for idx in indices]
        
        return experiences, indices, weights
        
    def update_priorities(self, indices, priorities):
        """Update priorities based on TD error"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.eps
            
    def __len__(self):
        return len(self.memory)