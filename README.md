# MyRLAlgos

My Implementations of standard Reinforcement Learning algorithms using PyTorch.   

The algorithms are implemented in [Pytorch](https://pytorch.org/) (without GPU support). The package uses `uv`. 

The algorithms are not implemented in a modular way. Each algorithm instead lives in it's own standalone python file. To run one, simply run `uv run <algorithm_name>`. For example, to run the Vanilla Policy Gradient algorithm, run `uv run VPG/vpg.py`.




### Algorithms -
- [X] Vanilla Policy Gradient (VPG)
- [X] Deep Q Network (DDQN)
- [ ] Deep Deterministic Policy Gradient (DDPG)
- [ ] Twin Delayed DDPG (TD3)
- [ ] Soft Actor Critic
- [ ] Proximal Policy Optimization
- [ ] Trust Region Policy Optimization


### References - 
[1] Spinning up documentation - https://spinningup.openai.com/en/latest/ - This is the main reference for the algorithms implemented here. The code is heavily inspired by the implementations provided in the spinning up documentation. 

[2] PyTorch documentation - https://pytorch.org/docs/stable/index.html - For understanding the PyTorch API used in the implementations. 

[3] OpenAI Gymnasium documentation - https://gymnasium.farama.org/ - For understanding the OpenAI Gymnasium API used in the implementations.
