import numpy as np
import torch
import torch.nn.functional as F
import yaml
import os
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from ppo import PPOAgent, PPONetwork
from dqn import DQNNetwork, DQNAgent

class CloudEdgeEnv:
    """
    Advanced cloud-edge environment for task scheduling with reinforcement learning.
    
    This environment models a system with edge devices and cloud servers,
    where tasks need to be scheduled for execution with realistic constraints.
    """
    
    def __init__(self, num_edge_devices=3, num_cloud_servers=2, max_steps=100):
        """
        Initialize the cloud-edge environment.
        
        Args:
            num_edge_devices: Number of edge devices in the system
            num_cloud_servers: Number of cloud servers available
            max_steps: Maximum steps per episode
        """
        self.num_edge_devices = num_edge_devices
        self.num_cloud_servers = num_cloud_servers
        self.max_steps = max_steps
        
        # State space dimensions
        self.state_dim = (
            num_edge_devices +  # Edge CPU utilization
            num_cloud_servers +  # Cloud CPU utilization
            1 +  # Network bandwidth
            num_edge_devices +  # Edge task queue
            num_cloud_servers +  # Cloud task queue
            1 +  # Task difficulty
            1 +  # Task data size
            1    # Task deadline
        )
        
        # Action space: 0 = process on edge device i, 1 = offload to cloud server j
        self.action_dim = num_edge_devices + num_cloud_servers
        
        # Initialize the environment
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            np.ndarray: Initial state representation
        """
        # Edge device CPU utilization (0-100%)
        self.edge_cpu = np.random.uniform(20, 60, self.num_edge_devices)
        
        # Cloud server CPU utilization (0-100%)
        self.cloud_cpu = np.random.uniform(30, 70, self.num_cloud_servers)
        
        # Network bandwidth between edge and cloud (Mbps)
        self.bandwidth = np.random.uniform(50, 150)
        
        # Task queues
        self.edge_queue = np.random.randint(0, 5, self.num_edge_devices)
        self.cloud_queue = np.random.randint(0, 8, self.num_cloud_servers)
        
        # Current task parameters
        self.task_difficulty = np.random.randint(1, 6)  # Computational complexity
        self.task_data_size = np.random.randint(1, 10)   # Data size in MB
        self.task_deadline = np.random.randint(8, 16)    # Deadline in time units
        
        # Track metrics
        self.total_time = 0
        self.energy_consumed = 0
        self.tasks_completed = 0
        self.deadlines_met = 0
        
        # Step counter
        self.steps = 0
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation.
        
        Returns:
            np.ndarray: State vector
        """
        # Normalize all values to [0,1] range
        return np.concatenate([
            self.edge_cpu / 100,                 # Edge CPU utilization (normalized)
            self.cloud_cpu / 100,                # Cloud CPU utilization (normalized)
            [self.bandwidth / 200],              # Network bandwidth (normalized)
            self.edge_queue / 10,                # Edge task queue (normalized)
            self.cloud_queue / 10,               # Cloud task queue (normalized)
            [self.task_difficulty / 10],         # Task difficulty (normalized)
            [self.task_data_size / 20],          # Task data size (normalized)
            [self.task_deadline / 15]            # Task deadline (normalized)
        ])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take an action in the environment.
        
        Args:
            action: Index indicating where to process task:
                   0 to num_edge_devices-1: Process on edge device
                   num_edge_devices to action_dim-1: Process on cloud server
        
        Returns:
            tuple: (next_state, reward, done, info)
        """
        self.steps += 1
        done = (self.steps >= self.max_steps)  # Episode ends after max_steps
        
        # Track execution time and energy consumption
        processing_time = 0
        energy = 0
        deadline_met = False
        info = {}
        
        # Process on edge
        if action < self.num_edge_devices:
            edge_idx = action
            
            # Calculate processing time based on edge CPU, task difficulty, and queue
            load_factor = 1 + (self.edge_cpu[edge_idx] / 100)
            queue_delay = self.edge_queue[edge_idx] * 0.5
            processing_time = (self.task_difficulty * load_factor) + queue_delay
            
            # Calculate energy consumption (edge is more energy efficient for small tasks)
            energy = self.task_difficulty * 0.7 * (1 + (self.task_data_size / 40))
            
            # Update edge CPU utilization and queue
            self.edge_cpu[edge_idx] += self.task_difficulty * 5
            self.edge_cpu[edge_idx] = min(self.edge_cpu[edge_idx], 100)
            self.edge_queue[edge_idx] = max(0, self.edge_queue[edge_idx] - 1) + 1
            
            info["location"] = f"edge_{edge_idx}"
            
        # Offload to cloud
        else:
            cloud_idx = action - self.num_edge_devices
            
            # Calculate network transfer time based on bandwidth and data size
            transfer_time = self.task_data_size * (100 / self.bandwidth)
            
            # Calculate cloud processing time (faster but depends on cloud CPU load and queue)
            load_factor = 1 + (self.cloud_cpu[cloud_idx] / 150)  # Cloud is faster
            queue_delay = self.cloud_queue[cloud_idx] * 0.3
            cloud_time = (self.task_difficulty * 0.6 * load_factor) + queue_delay
            
            # Total time includes transfer and cloud processing
            processing_time = transfer_time + cloud_time
            
            # Calculate energy (includes network transfer energy)
            energy = self.task_data_size * 0.8 + self.task_difficulty * 0.5
            
            # Update cloud CPU utilization and queue
            self.cloud_cpu[cloud_idx] += self.task_difficulty * 3
            self.cloud_cpu[cloud_idx] = min(self.cloud_cpu[cloud_idx], 100)
            self.cloud_queue[cloud_idx] = max(0, self.cloud_queue[cloud_idx] - 1) + 1
            
            info["location"] = f"cloud_{cloud_idx}"
            info["transfer_time"] = transfer_time
            info["cloud_time"] = cloud_time
        
        # Check if deadline was met
        deadline_met = processing_time <= self.task_deadline
        
        # Update metrics
        self.total_time += processing_time
        self.energy_consumed += energy
        self.tasks_completed += 1
        if deadline_met:
            self.deadlines_met += 1
        
        # Calculate reward
        time_penalty = -processing_time * 0.5
        energy_penalty = -energy * 0.02
        deadline_reward = 25.0 if deadline_met else -10.0
        reward = (time_penalty + energy_penalty + deadline_reward) 

        if processing_time > 15:
            reward -= 3.0

        reward = max(min(reward, 25), -25)

        # Add info
        info["processing_time"] = processing_time
        info["energy"] = energy
        info["reward"] = reward
        info["deadline_met"] = deadline_met
        
        # Simulate natural resource usage reduction
        self._update_system_state()
        
        # Generate new task
        self._generate_new_task()
        
        return self._get_state(), reward, done, info
    
    def _update_system_state(self):
        """Update system state after task execution"""
        # Reduce CPU load slightly on all resources
        self.edge_cpu = np.maximum(self.edge_cpu - np.random.uniform(0, 3, self.num_edge_devices), 10)
        self.cloud_cpu = np.maximum(self.cloud_cpu - np.random.uniform(0, 2, self.num_cloud_servers), 20)
        
        # Update network conditions
        bandwidth_change = np.random.uniform(-10, 10)
        self.bandwidth = np.clip(self.bandwidth + bandwidth_change, 30, 200)
        
        # Process tasks in queues
        self.edge_queue = np.maximum(self.edge_queue - np.random.randint(0, 2, self.num_edge_devices), 0)
        self.cloud_queue = np.maximum(self.cloud_queue - np.random.randint(0, 2, self.num_cloud_servers), 0)
    
    def _generate_new_task(self):
        """Generate a new task with random parameters"""
        self.task_difficulty = np.random.randint(1, 11)
        self.task_data_size = np.random.randint(1, 21)
        self.task_deadline = np.random.randint(5, 16)
    
    def get_metrics(self) -> Dict:
        """
        Get performance metrics.
        
        Returns:
            Dict: Performance metrics for the episode
        """
        return {
            "total_time": self.total_time,
            "energy_consumed": self.energy_consumed,
            "tasks_completed": self.tasks_completed,
            "deadlines_met": self.deadlines_met,
            "deadline_success_rate": self.deadlines_met / max(1, self.tasks_completed),
            "avg_time_per_task": self.total_time / max(1, self.tasks_completed),
            "avg_energy_per_task": self.energy_consumed / max(1, self.tasks_completed)
        }
    
    def render(self):
        """Display current environment state"""
        print(f"Step: {self.steps}")
        print(f"Edge CPU: {self.edge_cpu}")
        print(f"Cloud CPU: {self.cloud_cpu}")
        print(f"Bandwidth: {self.bandwidth:.2f} Mbps")
        print(f"Edge Queues: {self.edge_queue}")
        print(f"Cloud Queues: {self.cloud_queue}")
        print(f"Task Difficulty: {self.task_difficulty}")
        print(f"Task Data Size: {self.task_data_size} MB")
        print(f"Task Deadline: {self.task_deadline} time units")
        print(f"Tasks Completed: {self.tasks_completed}")
        print(f"Deadlines Met: {self.deadlines_met}")
        print(f"Total Time: {self.total_time:.2f}")
        print(f"Total Energy: {self.energy_consumed:.2f}")
        print("-" * 40)

class CloudEdgeSimulator:
    """
    Simulator class for running and comparing RL algorithms on cloud-edge environments.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the simulator.
        
        Args:
            config_path: Path to configuration file (yaml)
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Create environment
        self.env = CloudEdgeEnv(
            num_edge_devices=self.config.get('num_edge_devices', 3),
            num_cloud_servers=self.config.get('num_cloud_servers', 2),
            max_steps=self.config.get('max_steps', 100)
        )
        
        # Initialize state and action dimensions
        self.state_dim = len(self.env.reset())
        self.action_dim = self.env.action_dim
        
        # Create agents
        self.dqn_agent = None
        self.ppo_agent = None
        
        # Track results
        self.dqn_results = []
        self.ppo_results = []
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f).get('cloud-edge', {})
        else:
            # Default configuration
            return {
                'num_edge_devices': 3,
                'num_cloud_servers': 2,
                'max_steps': 100,
                'replay_memory_size': 100000,
                'mini_batch_size': 32,
                'epsilon_init': 1.0,
                'epsilon_decay': 0.99995,
                'epsilon_min': 0.05,
                'learning_rate_a': 0.0001,
                'discount_factor_g': 0.99,
                'target_update_freq': 1000,
                'policy_lr': 0.0003,
                'value_lr': 0.001,
                'clip_ratio': 0.2,
                'entropy_coef': 0.01,
                'value_coef': 0.5,
                'gae_lambda': 0.95,
                'epochs': 10
            }
    
    def initialize_agents(self):
        """Initialize DQN and PPO agents"""
        # Initialize DQN agent
        self.dqn_agent = DQNAgent(self.state_dim, self.action_dim, self.config)
        
        # Initialize PPO agent
        self.ppo_agent = PPOAgent(self.state_dim, self.action_dim, self.config)
    
    def train_dqn(self, episodes=1000, update_frequency=4, log_frequency=10, save_frequency=100, save_path='models/dqn_cloud_edge.pt'):
        """
        Train DQN agent.
        
        Args:
            episodes: Number of episodes to train
            update_frequency: How often to update the network
            log_frequency: How often to log results
            save_frequency: How often to save the model
            save_path: Path to save the model
            
        Returns:
            list: Episode rewards
        """
        if self.dqn_agent is None:
            self.initialize_agents()
        
        # Create directory for saving model if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        rewards = []
        best_reward = -float('inf')
        step_counter = 0
        
        # Training loop
        for episode in range(1, episodes + 1):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Select action
                action = self.dqn_agent.select_action(state)
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                self.dqn_agent.store_experience(state, action, reward, next_state, done)
                
                # Update model if needed
                step_counter += 1
                if step_counter % update_frequency == 0:
                    self.dqn_agent.update_model()
                
                # Update state and reward
                state = next_state
                episode_reward += reward
            
            # Update epsilon for exploration
            self.dqn_agent.update_epsilon()
            
            # Store episode reward
            rewards.append(episode_reward)
            self.dqn_agent.reward_history.append(episode_reward)
            
            # Log progress
            if episode % log_frequency == 0:
                avg_reward = sum(rewards[-log_frequency:]) / log_frequency
                print(f"DQN Episode {episode}/{episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {self.dqn_agent.epsilon:.4f}")
                
                # Get environment metrics
                metrics = self.env.get_metrics()
                print(f"Deadline Success Rate: {metrics['deadline_success_rate']:.2f} | Avg Time: {metrics['avg_time_per_task']:.2f} | Avg Energy: {metrics['avg_energy_per_task']:.2f}")
            
            # Save model if it's the best so far
            if episode % save_frequency == 0:
                avg_reward = sum(rewards[-save_frequency:]) / save_frequency
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self.dqn_agent.save_model(save_path)
                    print(f"DQN Model saved with avg reward: {avg_reward:.2f}")
        
        # Save final model
        final_path = save_path.replace('.pt', '_final.pt')
        self.dqn_agent.save_model(final_path)
        print(f"DQN Final model saved at: {final_path}")
        
        return rewards
    
    def train_ppo(self, episodes=1000, steps_per_update=2048, log_frequency=10, save_frequency=100, save_path='models/ppo_cloud_edge.pt'):
        """
        Train PPO agent.
        
        Args:
            episodes: Number of episodes to train
            steps_per_update: Number of steps to collect before updating the policy
            log_frequency: How often to log results
            save_frequency: How often to save the model
            save_path: Path to save the model
            
        Returns:
            list: Episode rewards
        """
        if self.ppo_agent is None:
            self.initialize_agents()
        
        # Create directory for saving model if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        rewards = []
        best_reward = -float('inf')
        step_counter = 0
        
        # Training loop
        for episode in range(1, episodes + 1):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Select action
                action, log_prob, value = self.ppo_agent.select_action(state)
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                self.ppo_agent.store_experience(state, action, reward, value, log_prob, done)
                
                # Update state and reward
                state = next_state
                episode_reward += reward
                step_counter += 1
                
                # Update model if enough steps have been collected
                if step_counter % steps_per_update == 0:
                    actor_loss, critic_loss, entropy = self.ppo_agent.update_model()
                    print(f"PPO Update | Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f} | Entropy: {entropy:.4f}")
            
            # Store episode reward
            rewards.append(episode_reward)
            self.ppo_agent.reward_history.append(episode_reward)
            
            # Log progress
            if episode % log_frequency == 0:
                avg_reward = sum(rewards[-log_frequency:]) / log_frequency
                print(f"PPO Episode {episode}/{episodes} | Avg Reward: {avg_reward:.2f}")
                
                # Get environment metrics
                metrics = self.env.get_metrics()
                print(f"Deadline Success Rate: {metrics['deadline_success_rate']:.2f} | Avg Time: {metrics['avg_time_per_task']:.2f} | Avg Energy: {metrics['avg_energy_per_task']:.2f}")
            
            # Save model if it's the best so far
            if episode % save_frequency == 0:
                avg_reward = sum(rewards[-save_frequency:]) / save_frequency
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self.ppo_agent.save_model(save_path)
                    print(f"PPO Model saved with avg reward: {avg_reward:.2f}")
                
                # If end of episode, also update the model if there's remaining data
                if len(self.ppo_agent.states) > 0:
                    actor_loss, critic_loss, entropy = self.ppo_agent.update_model()
        
        # Save final model
        final_path = save_path.replace('.pt', '_final.pt')
        self.ppo_agent.save_model(final_path)
        print(f"PPO Final model saved at: {final_path}")
        
        return rewards
    
    def evaluate_agent(self, agent_type='dqn', episodes=100):
        """
        Evaluate trained agent.
        
        Args:
            agent_type: Type of agent to evaluate ('dqn' or 'ppo')
            episodes: Number of episodes to evaluate
            
        Returns:
            dict: Evaluation metrics
        """
        agent = self.dqn_agent if agent_type == 'dqn' else self.ppo_agent
        
        if agent is None:
            print(f"Error: {agent_type.upper()} agent not initialized")
            return {}
        
        rewards = []
        metrics_list = []
        
        # Evaluation loop
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Select action
                if agent_type == 'dqn':
                    action = agent.select_action(state, evaluation=True)
                else:
                    action, _, _ = agent.select_action(state, evaluation=True)
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Update state and reward
                state = next_state
                episode_reward += reward
            
            # Store episode reward and metrics
            rewards.append(episode_reward)
            metrics_list.append(self.env.get_metrics())
        
        # Calculate average metrics
        avg_reward = sum(rewards) / episodes
        avg_metrics = {
            'avg_reward': avg_reward,
            'deadline_success_rate': sum(m['deadline_success_rate'] for m in metrics_list) / episodes,
            'avg_time_per_task': sum(m['avg_time_per_task'] for m in metrics_list) / episodes,
            'avg_energy_per_task': sum(m['avg_energy_per_task'] for m in metrics_list) / episodes
        }
        
        print(f"\n{agent_type.upper()} Evaluation Results ({episodes} episodes):")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Deadline Success Rate: {avg_metrics['deadline_success_rate']:.2f}")
        print(f"Average Time per Task: {avg_metrics['avg_time_per_task']:.2f}")
        print(f"Average Energy per Task: {avg_metrics['avg_energy_per_task']:.2f}")
        
        return avg_metrics
    
    def compare_algorithms(self, training_episodes=500, eval_episodes=100):
        """
        Train and compare DQN and PPO algorithms.
        
        Args:
            training_episodes: Number of episodes to train each agent
            eval_episodes: Number of episodes to evaluate each agent
            
        Returns:
            dict: Comparison results
        """
        # Initialize agents
        self.initialize_agents()
        
        # Train DQN
        print("\nTraining DQN agent...")
        dqn_rewards = self.train_dqn(episodes=training_episodes)
        
        # Train PPO
        print("\nTraining PPO agent...")
        ppo_rewards = self.train_ppo(episodes=training_episodes)
        
        # Evaluate agents
        print("\nEvaluating agents...")
        dqn_metrics = self.evaluate_agent(agent_type='dqn', episodes=eval_episodes)
        ppo_metrics = self.evaluate_agent(agent_type='ppo', episodes=eval_episodes)
        
        # Compare results
        print("\nComparison Results:")
        print(f"DQN vs PPO Average Reward: {dqn_metrics['avg_reward']:.2f} vs {ppo_metrics['avg_reward']:.2f}")
        print(f"DQN vs PPO Deadline Success: {dqn_metrics['deadline_success_rate']:.2f} vs {ppo_metrics['deadline_success_rate']:.2f}")
        print(f"DQN vs PPO Avg Time: {dqn_metrics['avg_time_per_task']:.2f} vs {ppo_metrics['avg_time_per_task']:.2f}")
        print(f"DQN vs PPO Avg Energy: {dqn_metrics['avg_energy_per_task']:.2f} vs {ppo_metrics['avg_energy_per_task']:.2f}")
        
        # Plot learning curves
        self.plot_learning_curves(dqn_rewards, ppo_rewards)
        
        return {
            'dqn': {
                'rewards': dqn_rewards,
                'metrics': dqn_metrics
            },
            'ppo': {
                'rewards': ppo_rewards,
                'metrics': ppo_metrics
            }
        }
    
    def plot_learning_curves(self, dqn_rewards, ppo_rewards):
        """
        Plot learning curves for DQN and PPO.
        
        Args:
            dqn_rewards: List of DQN episode rewards
            ppo_rewards: List of PPO episode rewards
        """
        # Smooth rewards for better visualization
        def smooth(data, window=10):
            return [sum(data[max(0, i-window):i+1]) / min(window, i+1) for i in range(len(data))]
        
        smooth_dqn = smooth(dqn_rewards)
        smooth_ppo = smooth(ppo_rewards)
        
        plt.figure(figsize=(12, 6))
        
        # Plot episode rewards
        plt.subplot(1, 2, 1)
        plt.plot(smooth_dqn, label='DQN')
        plt.plot(smooth_ppo, label='PPO')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Learning Curves')
        plt.legend()
        
        # Plot loss curves
        plt.subplot(1, 2, 2)
        if self.dqn_agent and self.dqn_agent.loss_history:
            smooth_loss = smooth(self.dqn_agent.loss_history)
            plt.plot(smooth_loss, label='DQN Loss')
        if self.ppo_agent and self.ppo_agent.actor_loss_history:
            smooth_actor_loss = smooth(self.ppo_agent.actor_loss_history)
            plt.plot(smooth_actor_loss, label='PPO Actor Loss')
        plt.xlabel('Update')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('learning_curves.png')
        plt.show()


# EdgeCloudSim Integration Helper
def create_edge_cloud_sim_connector(model_path, agent_type='dqn', state_dim=None, action_dim=None):
    """
    Create a connector class that can be used from Java EdgeCloudSim to make predictions.
    
    Args:
        model_path: Path to the saved model
        agent_type: Type of agent ('dqn' or 'ppo')
        state_dim: State dimension
        action_dim: Action dimension
        
    Returns:
        function: Prediction function that can be called from Java
    """
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if agent_type == 'dqn':
        # Create model
        model = DQNNetwork(state_dim, action_dim)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['policy_net'])
        model.eval()
        
        # Create prediction function
        def predict(state):
            """
            Make prediction using DQN model.
            
            Args:
                state: State vector as numpy array
                
            Returns:
                int: Selected action
            """
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                return q_values.argmax().item()
    
    else:  # PPO
        # Create model
        model = PPONetwork(state_dim, action_dim)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Create prediction function
        def predict(state):
            """
            Make prediction using PPO model.
            
            Args:
                state: State vector as numpy array
                
            Returns:
                int: Selected action
            """
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action_logits, _ = model(state_tensor)
                action_probs = F.softmax(action_logits, dim=-1)
                return torch.argmax(action_probs, dim=-1).item()
    
    return predict