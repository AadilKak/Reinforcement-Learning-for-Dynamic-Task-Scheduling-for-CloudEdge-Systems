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

class PPONetwork(nn.Module):
    """
    Actor-Critic network for PPO algorithm.
    """
    
    def __init__(self, input_dim, output_dim, shared_layers=True):
        super(PPONetwork, self).__init__()
        
        # Feature extraction layers (shared between actor and critic)
        self.shared_layers = shared_layers
        if shared_layers:
            self.feature_layer = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )
            
            # Actor: policy network
            self.actor_head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            )
            
            # Critic: value network
            self.critic_head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        else:
            # Separate networks for actor and critic
            self.actor = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            )
            
            self.critic = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            tuple: (action_logits, value)
        """
        if self.shared_layers:
            features = self.feature_layer(x)
            action_logits = self.actor_head(features)
            value = self.critic_head(features)
        else:
            action_logits = self.actor(x)
            value = self.critic(x)
            
        return action_logits, value
    
    def get_action_probs(self, x):
        """
        Get action probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Action probabilities
        """
        if self.shared_layers:
            features = self.feature_layer(x)
            action_logits = self.actor_head(features)
        else:
            action_logits = self.actor(x)
            
        return F.softmax(action_logits, dim=-1)
    
    def get_value(self, x):
        """
        Get value estimate.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Value estimate
        """
        if self.shared_layers:
            features = self.feature_layer(x)
            value = self.critic_head(features)
        else:
            value = self.critic(x)
            
        return value


class PPOAgent:
    """
    Proximal Policy Optimization agent for cloud-edge task scheduling.

    Implements PPO algorithm with clipped surrogate objective, 
    entropy bonus, and generalized advantage estimation.
    """
    def __init__(self, state_dim, action_dim, config):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Dictionary containing hyperparameters
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Extract hyperparameters from config
        self.gamma = config.get('discount_factor_g', 0.99)
        self.policy_lr = config.get('policy_lr', 0.0003)
        self.value_lr = config.get('value_lr', 0.001)
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.epochs = config.get('epochs', 10)
        self.batch_size = config.get('mini_batch_size', 64)
        
        # Create PPO network
        self.network = PPONetwork(state_dim, action_dim)
        
        # Create optimizers
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.policy_lr)
        
        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        
        # Create buffers for storing experience
        self.reset_buffers()
        
        # Stats
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.entropy_history = []
        self.reward_history = []
        
    def reset_buffers(self):
        """Reset experience buffers"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def select_action(self, state, evaluation=False):
        """
        Select action using policy network.
        
        Args:
            state: Current state
            evaluation: If True, use greedy policy (no exploration)
            
        Returns:
            tuple: (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits, value = self.network(state_tensor)
            
            # Get action probabilities
            action_probs = F.softmax(action_logits, dim=-1)
            
            # Sample action
            if evaluation:
                # Greedy action selection
                action = torch.argmax(action_probs, dim=-1)
            else:
                # Sample from action distribution
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                
            # Calculate log probability
            log_prob = torch.log(action_probs.squeeze(0)[action.item()] + 1e-8)
        
        return action.item(), log_prob.item(), value.item()

    def store_experience(self, state, action, reward, value, log_prob, done):
        """Store experience in buffers"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def update_model(self):
        """Update model using experiences from buffers"""
        # Check if we have enough data
        if len(self.states) == 0:
            return 0.0, 0.0, 0.0
        
        # Convert lists to tensors
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)
        values = torch.FloatTensor(self.values).to(self.device)
        
        # Calculate advantages using Generalized Advantage Estimation (GAE)
        advantages = []
        gae = 0
        with torch.no_grad():
            # Get values for the last states
            next_values = self.network.get_value(states)
            
            # Calculate deltas and advantages
            for i in reversed(range(len(rewards))):
                if i == len(rewards) - 1:
                    next_val = 0 if dones[i] else next_values[i].item()
                else:
                    next_val = values[i+1]
                
                delta = rewards[i] + self.gamma * next_val * (1 - dones[i]) - values[i]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
                advantages.insert(0, gae)
                
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Calculate returns (advantage + value)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Prepare dataset for minibatch training
        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, advantages, returns)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Track losses for this update
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        # Perform multiple epochs of training
        for _ in range(self.epochs):
            for batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in dataloader:
                # Get current action probabilities and values
                action_logits, values = self.network(batch_states)
                action_probs = F.softmax(action_logits, dim=-1)
                
                # Create distribution and calculate log probabilities
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                
                # Calculate entropy for exploration bonus
                entropy = dist.entropy().mean()
                
                # Calculate ratio for PPO clipping
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Calculate surrogate losses
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                
                # Calculate actor loss (negative because we're trying to maximize the objective)
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Calculate critic loss
                critic_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Calculate total loss
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                
                # Update the network
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                
                # Record losses
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
        
        # Calculate average losses
        avg_actor_loss = total_actor_loss / (self.epochs * len(dataloader))
        avg_critic_loss = total_critic_loss / (self.epochs * len(dataloader))
        avg_entropy = total_entropy / (self.epochs * len(dataloader))
        
        # Store losses for tracking
        self.actor_loss_history.append(avg_actor_loss)
        self.critic_loss_history.append(avg_critic_loss)
        self.entropy_history.append(avg_entropy)
        
        # Clear experience buffers
        self.reset_buffers()
        
        return avg_actor_loss, avg_critic_loss, avg_entropy

    def save_model(self, path):
        """Save model weights to file"""
        torch.save(self.network.state_dict(), path)
        
    def load_model(self, path):
        """Load model weights from file"""
        self.network.load_state_dict(torch.load(path))
