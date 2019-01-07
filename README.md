[//]: # (Image References)

[image1]: trained_agent.gif "Trained Agent"

# Double DQN Banana Collector

### Introduction

This repo hosts an implementation of Double DQN which can solve the [modified Unity ml-agents environment](https://github.com/Unity-Technologies/ml-agents) 
utilized by the Udacity Deep Reinforcement Learning Nanodegree course. Also provided are the trained weights, which 
score an average of +15.0 to +16.0 across 100 episodes.

![Trained Agent][image1]

### Rewards and Goal
The goal is for an agent to earn more than +13 points per episode of the game. An agent receives +1 point for collecting 
a yellow banana and -1 point for collecting a blue banana. There are no other rewards.

### Game Environment
(from the [DRLND README](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation))

    The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects 
    around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four 
    discrete actions are available, corresponding to:
    
    - 0 - move forward.
    - 1 - move backward.
    - 2 - turn left.
    - 3 - turn right.

### Setup
Simply follow the setup instructions from the [DRLND README](https://github.com/udacity/deep-reinforcement-learning). No 
additional steps should be required

### Instructions

A number of command-line arguments have been defined. See the `parse_arguments()` function in `banana.py` or run:  
> `python banana.py --help`

#### Training an Agent
To train an agent:  
> `python banana.py`

Training an agent with some modified hyper-parameters:
> `python banana.py --episodes 2000 --eps 0.75 --gamma 0.9`

Resume training an agent whose weights have been saved as `checkpoint.pth`:
> `python banana.py --checkpoint checkpoint.pth`

#### Testing an Agent
Running an agent with the `--evaluate` flag will not perform any learning. It will still track the scores for each 
episode and save them to a text file for future analysis. Using the `--slow` flag will run the game at normal speed (so 
you can see what's happening!).  
> `python banana.py --checkpoint checkpoint.pth --evaluate --slow --episodes 100 --eps 0 --eps-min 0`

#### Report
See `report.md` for further implementation details and performance. 