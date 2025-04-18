from cloud_egde_agent import CloudEdgeSimulator
import matplotlib.pyplot as plt
# Load hyperparameters and set up environment
simulator = CloudEdgeSimulator(config_path="hyperparameters.yaml")

# Train using DQN or PPO
dqn_rewards = simulator.train_dqn(episodes=1000)  # You can change episodes to 500+
#ppo_rewards = simulator.train_ppo(episodes=300)

# Optionally evaluate
metrics = simulator.evaluate_agent(agent_type='dqn', episodes=50)
print(metrics)

plt.plot(simulator.dqn_agent.reward_history, label='DQN')
#plt.plot(simulator.ppo_agent.reward_history, label='PPO')
plt.legend()
plt.title("Episode Rewards")
plt.show()

# Compare both if you like
# simulator.compare_algorithms(training_episodes=300, eval_episodes=50)
