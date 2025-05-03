import argparse
from cloud_egde_agent import CloudEdgeSimulator
import matplotlib.pyplot as plt

"""
def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained DQN and PPO models on Cloud-Edge simulator")
    parser.add_argument('--dqn_model', type=str, required=True, help='Path to the saved DQN model .pt file')
    parser.add_argument('--ppo_model', type=str, required=True, help='Path to the saved PPO model .pt file')
    parser.add_argument('--episodes', type=int, default=500, help='Number of evaluation episodes')
    parser.add_argument('--config', type=str, default='hyperparameters.yaml', help='Path to hyperparameters.yaml')
    args = parser.parse_args()

    # Initialize simulator and agents
    sim = CloudEdgeSimulator(config_path=args.config)
    sim.initialize_agents()

    # Load pretrained weights
    print(f"Loading DQN model from {args.dqn_model}...")
    sim.dqn_agent.load_model(args.dqn_model)
    print(f"Loading PPO model from {args.ppo_model}...")
    sim.ppo_agent.load_model(args.ppo_model)

    # Evaluate each agent
    print(f"Evaluating DQN for {args.episodes} episodes...")
    dqn_metrics = sim.evaluate_agent(agent_type='dqn', episodes=args.episodes)
    print(f"Evaluating PPO for {args.episodes} episodes...")
    ppo_metrics = sim.evaluate_agent(agent_type='ppo', episodes=args.episodes)

    # Display metrics
    print("\n=== Evaluation Results ===")
    print(f"Metric                  |  DQN       |  PPO")
    print("-------------------------+------------+------------")
    print(f"Average Reward          | {dqn_metrics['avg_reward']:<10.2f} | {ppo_metrics['avg_reward']:<10.2f}")
    print(f"Deadline Success Rate   | {dqn_metrics['deadline_success_rate']:<10.2f} | {ppo_metrics['deadline_success_rate']:<10.2f}")
    print(f"Avg Time per Task       | {dqn_metrics['avg_time_per_task']:<10.2f} | {ppo_metrics['avg_time_per_task']:<10.2f}")
    print(f"Avg Energy per Task     | {dqn_metrics['avg_energy_per_task']:<10.2f} | {ppo_metrics['avg_energy_per_task']:<10.2f}")

    # Plot comparison bar chart
    labels = ['Avg Reward', 'Success Rate', 'Time/Task', 'Energy/Task']
    dqn_vals = [dqn_metrics['avg_reward'], dqn_metrics['deadline_success_rate'],
                dqn_metrics['avg_time_per_task'], dqn_metrics['avg_energy_per_task']]
    ppo_vals = [ppo_metrics['avg_reward'], ppo_metrics['deadline_success_rate'],
                ppo_metrics['avg_time_per_task'], ppo_metrics['avg_energy_per_task']]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width/2 for i in x], dqn_vals, width, label='DQN')
    ax.bar([i + width/2 for i in x], ppo_vals, width, label='PPO')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title(f'Comparison over {args.episodes} Episodes')
    ax.legend()
    plt.tight_layout()
    plt.savefig('evaluation_comparison.png')
    print("Saved comparison plot to evaluation_comparison.png")
    plt.show()

if __name__ == '__main__':
    main()
"""