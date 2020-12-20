---
title: Efficient Exploration via Actor-Critic Ensemble
published: True
keywords: [Reinforcement Learning]
---

This is a conclusion of my final project for CS 287: Advanced Robotics at UC Berkeley. For more information, please look at our [PDF report](/assets/CS287_Report.pdf) and [slides](/assets/CS287_Presentation.pdf).

Off-policy actor-critic Reinforcement Learning (RL) algorithms like Deep Deterministic Policy Gradient (DDPG) suffer from instability and dependence on careful hyperparameter tuning.

We proposed an off-policy actor-critic RL algorithm, Full Ensemble Deep Deterministic Policy Gradient (FEDDPG) that uses two ensemble functions to combine multiple actor networks and critic networks for exploration.

In addition to using an ensemble, our algorithm improved the stability and robustness of DDPG by incorporating multi-step learning and prioritized experience replay. The performance of the agent outperformed state-of-the-art off-policy methods on multiple MuJoCo continuous control environments. 

We also described an ensemble formulation of SAC, indicating that some of our approaches can also be applied to algorithms that learn a non-deterministic policy.
