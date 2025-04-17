import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from experience_replay import EnhancedReplayMemory, Experience

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
    
class DQNAgent:
    """
    Deep Q-Network agent for cloud-edge task scheduling.
    
    Implements Double DQN with prioritized experience replay,
    dueling architecture, and noisy networks for exploration.
    """
    
    def __init__(self, state_dim, action_dim, config):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Dictionary containing hyperparameters
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Extract hyperparameters from config
        self.gamma = config.get('discount_factor_g', 0.99)
        self.lr = config.get('learning_rate_a', 0.0001)
        self.epsilon = config.get('epsilon_init', 1.0)
        self.epsilon_decay = config.get('epsilon_decay', 0.99995)
        self.epsilon_min = config.get('epsilon_min', 0.05)
        self.target_update_freq = config.get('target_update_freq', 1000)
        self.batch_size = config.get('mini_batch_size', 32)
        
        # Create Q networks
        self.policy_net = DQNNetwork(state_dim, action_dim)
        self.target_net = DQNNetwork(state_dim, action_dim)
        
        # Initialize target network with policy network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode
        
        # Create optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Create replay memory
        memory_size = config.get('replay_memory_size', 100000)
        self.memory = EnhancedReplayMemory(memory_size)
        
        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        
        # Track updates
        self.update_counter = 0
        
        # Stats
        self.loss_history = []
        self.reward_history = []
        
    def select_action(self, state, evaluation=False):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            evaluation: If True, use greedy policy (no exploration)
            
        Returns:
            int: Selected action
        """
        if evaluation or random.random() > self.epsilon:
            # Greedy action selection
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
        else:
            # Random action selection
            return random.randint(0, self.action_dim - 1)
    
    def update_epsilon(self):
        """Decay epsilon for exploration"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.push(state, action, reward, next_state, done)
    
    def update_model(self):
        """Update model using batch of experiences from replay memory"""
        if len(self.memory) < self.batch_size:
            return 0.0  # Not enough samples
        
        # Sample batch of experiences
        experiences, indices, weights = self.memory.sample(self.batch_size)
        
        if not experiences:
            return 0.0
        
        # Extract experiences
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)
        weights_tensor = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        # Calculate current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Calculate target Q values using Double DQN
        # 1. Select action using policy network
        next_actions = self.policy_net(next_state_batch).argmax(1, keepdim=True)
        # 2. Evaluate action using target network
        next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
        # 3. Calculate target Q values
        target_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
        
        # Calculate loss
        td_errors = torch.abs(target_q_values - current_q_values).detach().cpu().numpy()
        loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none') * weights_tensor
        loss = loss.mean()
        
        # Update priorities in replay memory
        self.memory.update_priorities(indices, td_errors)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent explosion
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()
        
        # Update target network if needed
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Store loss for tracking
        self.loss_history.append(loss.item())
        
        return loss.item()
    
    def save_model(self, path):
        """Save model weights to file"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        
    def load_model(self, path):
        """Load model weights from file"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']