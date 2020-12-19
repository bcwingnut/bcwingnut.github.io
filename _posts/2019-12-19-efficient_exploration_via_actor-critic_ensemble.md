---
title: Efficient Exploration via Actor-Critic Ensemble
published: True
---

## Abstract

Off-policy actor-critic Reinforcement Learning (RL) algorithms like Deep Deterministic Policy Gradient (DDPG) suffer from instability and dependence on careful hyperparameter tuning. We propose an off-policy actor-critic RL algorithm, Full Ensemble Deep Deterministic Policy Gradient (FEDDPG), that uses two ensemble functions to combine multiple actor networks and critic networks for exploration. In addition to using an ensemble, our algorithm improves the stability and robustness of DDPG by incorporating multi-step learning and prioritized experience replay. The performance of the agent outperforms state-of-the-art off-policy methods on multiple MuJoCo continuous control environments. Finally, we describe an ensemble formulation of SAC, indicating that some of our approaches can also be applied to algorithms that learn a non-deterministic policy.

## More

This is the report of my final project for CS 287: Advanced Robotics at UC Berkeley. Full text can be accessed [here](https://drive.google.com/file/d/16fIUajs7Ozp3NWCayg378s1HchDof7jN/view).
