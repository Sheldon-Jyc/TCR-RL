# TCR_RL
This is the code for the paper: Learning Long-Horizon Temporal Dependencies in Reinforcement Learning: A Selective State-Space and Continuous-Time Architecture for Robotic Control

## Introduction
This folder implements the core algorithm of SSM-LTC. The complete training code will be open source after review.

## Training Profile
The following one is the training profile of ssm-ltc, in which you can configure the hyperparameters such as the number of smooth ODE units, number of neural network layers and so on.
```bash
run.py
```

## SmODE implementation
In ```model\ssm_ltc.py```, we implement how to construct network as a policy network for reinforcement learning. 

In the ```algorithm\ppo``` folder, we have implemented ppo algorithms: ssm-ltc can be easily integrated with this RL methods.
