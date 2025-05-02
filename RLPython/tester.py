import numpy as np
from cloud_egde_agent import CloudEdgeSimulator
import matplotlib.pyplot as plt


def evaluate_agent_metrics(simulator, agent_type, episodes=100):
    """
    Run the specified agent on the environment for a number of episodes,
    recording per-episode reward, total processing time, deadline success rate, and average energy.
    """
    env = simulator.env
    agent = simulator.dqn_agent if agent_type == 'dqn' else simulator.ppo_agent

    rewards = []
    processing_times = []
    deadline_rates = []
    energies = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            if agent_type == 'dqn':
                action = agent.select_action(state, evaluation=True)
            else:
                action, _, _ = agent.select_action(state, evaluation=True)
            state, reward, done, info = env.step(action)
            ep_reward += reward

        # Record metrics after episode finishes
        metrics = env.get_metrics()
        rewards.append(ep_reward)
        processing_times.append(metrics['total_time'])
        deadline_rates.append(metrics['deadline_success_rate'])
        energies.append(metrics['avg_energy_per_task'])

    return rewards, processing_times, deadline_rates, energies


def main():
    # Initialize simulator and train
    simulator = CloudEdgeSimulator(config_path="hyperparameters.yaml")
    print("Training DQN...")
    dqn_rewards = simulator.train_dqn(episodes=500)
    print("Training PPO...")
    ppo_rewards = simulator.train_ppo(episodes=500)

    # Evaluate agents over a fixed number of episodes
    eval_episodes = 200
    dqn_data = evaluate_agent_metrics(simulator, 'dqn', episodes=eval_episodes)
    ppo_data = evaluate_agent_metrics(simulator, 'ppo', episodes=eval_episodes)
    x = np.arange(1, eval_episodes + 1)

    # Plot definitions
    plots = [
        (dqn_data[0], ppo_data[0], 'Total Reward', 'episode_reward.png'),
        (dqn_data[1], ppo_data[1], 'Total Processing Time', 'processing_time.png'),
        (dqn_data[2], ppo_data[2], 'Deadline Success Rate', 'deadline_success_rate.png'),
        (dqn_data[3], ppo_data[3], 'Avg Energy per Task', 'energy_per_task.png')
    ]

    for idx, (dqn_vals, ppo_vals, ylabel, filename) in enumerate(plots, 1):
        plt.figure()
        plt.plot(x, dqn_vals, label='DQN')
        plt.plot(x, ppo_vals, label='PPO')
        plt.xlabel('Episode')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} per Episode')
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
        plt.show()

    # Plot PPO/DQN loss curves
    simulator.plot_learning_curves(dqn_rewards, ppo_rewards)

if __name__ == '__main__':
    main()
