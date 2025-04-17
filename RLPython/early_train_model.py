from cloud_egde_agent import CloudEdgeSimulator

# Load hyperparameters and set up environment
simulator = CloudEdgeSimulator(config_path="hyperparameters.yaml")

# Train using DQN or PPO
# dqn_rewards = simulator.train_dqn(episodes=300)  # You can change episodes to 500+
ppo_rewards = simulator.train_ppo(episodes=300)

# Optionally evaluate
metrics = simulator.evaluate_agent(agent_type='dqn', episodes=50)
print(metrics)

# Compare both if you like
# simulator.compare_algorithms(training_episodes=300, eval_episodes=50)
