# AI-Driven Framework: Deep Reinforcement Learning for Dynamic Task Scheduling in Cloud-Edge Systems

**Authors:** Yohannes Geleta & Aadil Kakkidi  
**GitHub:** [https://github.com/yohannesgeleta/512Project](https://github.com/yohannesgeleta/512Project)  

---

## Overview
This project implements a deep reinforcement learning (DRL) framework for dynamic task scheduling in cloud-edge computing environments. The system learns optimal task allocation policies to minimize processing time, reduce energy consumption, and balance workloads across cloud and edge nodes.

## Motivation
Cloud-edge systems support low-latency, high-throughput applications like IoT and real-time analytics. Traditional scheduling methods struggle with dynamic workloads and heterogeneous resources. DRL offers a learning-based approach to optimize task scheduling without relying on predefined rules.

## Key Features
- **Deep Q-Network (DQN):** Value-based DRL with dueling architecture, experience replay, and ε-greedy exploration.  
- **Proximal Policy Optimization (PPO):** Policy-based DRL with shared actor-critic network, generalized advantage estimation, and clipped surrogate objective.  
- **EdgeCloudSim:** Simulates realistic cloud-edge environments including latency, mobility, and hierarchical architecture.  
- **Multi-objective Reward Design:** Optimizes task completion time, energy efficiency, and deadline compliance.

## Methodology
1. Implemented DQN and PPO for dynamic task scheduling.  
2. Simulated cloud-edge scenarios with EdgeCloudSim.  
3. Designed reward heuristics to optimize speed, energy, and deadlines.  
4. Evaluated performance across multiple episodes for consistency and efficiency.

## Results
- **DQN:** Higher average reward, better deadline success rate (~70%), lower task completion time (~8 units), moderate energy usage.  
- **PPO:** Lower reward but potentially better energy efficiency; results less consistent.  
- Peak performance sometimes hard to replicate under scaled conditions.

## Future Work
- Improve reward heuristics and model stability.  
- Explore alternative EdgeCloudSim applications (e.g., FuzzyNetwork).  
- Conduct further research to reach performance targets under larger workloads.

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yohannesgeleta/512Project.git
2. Install dependencies (Python, PyTorch, EdgeCloudSim, Java).

3. Run simulations and train DQN/PPO models.

4. Evaluate task scheduling performance under different cloud-edge conditions.

## Accompanying Paper

The full research paper associated with this project is:

Yohannes Geleta, An AI-Driven Framework: Deep Reinforcement Learning for Dynamic Task Scheduling and Load Balancing for Cloud-Edge Systems to Enhance Resource Utilization, Graduate Student (Computer Science), Penn State Harrisburg.
Email: yxg5342@psu.edu

Contribution: 50%

The paper presents:

Implementation of DQN and PPO RL models in EdgeCloudSim.

Experiments comparing RL-based schedulers to traditional task scheduling methods.

Detailed analysis of task completion, latency, CPU utilization, and QoE.

Discussion of failure modes and future improvements.

Access the full paper: Download PDF

References: [Full reference list in paper]
