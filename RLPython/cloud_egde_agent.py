import numpy as np
from dqn import DQN
from ppo import PPO
from experience_replay import ReplayMemory

class SimpleCloudEdgeEnv:
    """
    A simple cloud-edge environment for task scheduling with reinforcement learning.
    
    This environment models a system with edge devices and cloud servers,
    where tasks need to be scheduled for execution.
    """
    
    def __init__(self, num_edge_devices=3, num_cloud_servers=2):
        """
        Initialize the cloud-edge environment.
        
        Args:
            num_edge_devices: Number of edge devices in the system
            num_cloud_servers: Number of cloud servers available
        """
        self.num_edge_devices = num_edge_devices
        self.num_cloud_servers = num_cloud_servers
        
        # State space dimensions
        self.state_dim = (
            num_edge_devices +  # Edge CPU utilization
            num_cloud_servers +  # Cloud CPU utilization
            1 +  # Network bandwidth
            1    # Current task difficulty
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
        
        # Current task difficulty (1-10)
        self.task_difficulty = np.random.randint(1, 11)
        
        # Track metrics
        self.total_time = 0
        self.energy_consumed = 0
        self.tasks_completed = 0
        
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
            [self.task_difficulty / 10]          # Task difficulty (normalized)
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
        done = (self.steps >= 100)  # Episode length of 100 steps
        
        # Track execution time and energy consumption
        processing_time = 0
        energy = 0
        info = {}
        
        # Process on edge
        if action < self.num_edge_devices:
            edge_idx = action
            
            # Calculate processing time based on edge CPU and task difficulty
            load_factor = 1 + (self.edge_cpu[edge_idx] / 100)
            processing_time = self.task_difficulty * load_factor
            
            # Calculate energy consumption (edge is more energy efficient)
            energy = self.task_difficulty * 0.7
            
            # Update edge CPU utilization
            self.edge_cpu[edge_idx] += self.task_difficulty * 5
            self.edge_cpu[edge_idx] = min(self.edge_cpu[edge_idx], 100)
            
            info["location"] = f"edge_{edge_idx}"
            
        # Offload to cloud
        else:
            cloud_idx = action - self.num_edge_devices
            
            # Calculate network transfer time based on bandwidth
            transfer_time = self.task_difficulty * (100 / self.bandwidth)
            
            # Calculate cloud processing time (faster but depends on cloud CPU load)
            load_factor = 1 + (self.cloud_cpu[cloud_idx] / 150)  # Cloud is faster
            cloud_time = self.task_difficulty * 0.6 * load_factor
            
            # Total time includes transfer and cloud processing
            processing_time = transfer_time + cloud_time
            
            # Calculate energy (includes network transfer energy)
            energy = self.task_difficulty * 1.2
            
            # Update cloud CPU utilization
            self.cloud_cpu[cloud_idx] += self.task_difficulty * 3
            self.cloud_cpu[cloud_idx] = min(self.cloud_cpu[cloud_idx], 100)
            
            info["location"] = f"cloud_{cloud_idx}"
            info["transfer_time"] = transfer_time
            info["cloud_time"] = cloud_time
        
        # Update metrics
        self.total_time += processing_time
        self.energy_consumed += energy
        self.tasks_completed += 1
        
        # Calculate reward (negative time plus small energy efficiency component)
        reward = -processing_time - (energy * 0.1)
        
        # Add info
        info["processing_time"] = processing_time
        info["energy"] = energy
        info["reward"] = reward
        
        # Simulate natural resource usage reduction
        self._update_system_state()
        
        # Generate new task
        self.task_difficulty = np.random.randint(1, 11)
        
        return self._get_state(), reward, done, info
    
    def _update_system_state(self):
        """Update system state after task execution"""
        # Reduce CPU load slightly on all resources
        self.edge_cpu = np.maximum(self.edge_cpu - np.random.uniform(0, 3, self.num_edge_devices), 10)
        self.cloud_cpu = np.maximum(self.cloud_cpu - np.random.uniform(0, 2, self.num_cloud_servers), 20)
        
        # Update network conditions
        bandwidth_change = np.random.uniform(-10, 10)
        self.bandwidth = np.clip(self.bandwidth + bandwidth_change, 30, 200)
    
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
            "avg_time_per_task": self.total_time / max(1, self.tasks_completed),
            "avg_energy_per_task": self.energy_consumed / max(1, self.tasks_completed)
        }
    
    def render(self):
        """Display current environment state"""
        print(f"Step: {self.steps}")
        print(f"Edge CPU: {self.edge_cpu}")
        print(f"Cloud CPU: {self.cloud_cpu}")
        print(f"Bandwidth: {self.bandwidth:.2f} Mbps")
        print(f"Task Difficulty: {self.task_difficulty}")
        print(f"Tasks Completed: {self.tasks_completed}")
        print(f"Total Time: {self.total_time:.2f}")
        print(f"Total Energy: {self.energy_consumed:.2f}")
        print("-" * 40)


# Example usage
if __name__ == "__main__":
    # Create environment
    env = SimpleCloudEdgeEnv(num_edge_devices=2, num_cloud_servers=1)
    
    # Reset environment
    state = env.reset()
    print("Initial State:", state)
    
    # Take some random actions
    for i in range(10):
        action = np.random.randint(0, env.action_dim)
        next_state, reward, done, info = env.step(action)
        
        print(f"\nAction: {action} ({info['location']})")
        print(f"Reward: {reward:.2f}")
        env.render()
        
        if done:
            break
    
    # Show final metrics
    print("\nFinal Metrics:")
    metrics = env.get_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")