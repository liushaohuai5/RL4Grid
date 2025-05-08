# RL4Grid Environment

This is a custom reinforcement learning environment, designed for power system optimal dispatch problem.

## Features

- action: generator active power setpoints
- networks: IEEE 14, 39, 57, 300 systems and SG126

## Installation

### Install via `pip`

You can install the environment package using `pip`:

```bash
pip install RL4Grid
```

## Usage Example
Once installed, you can use the reinforcement learning environment as follows:

```python

import gym
import RL4Grid  # Import your environment

# Create the environment
env = gym.make("MyRL-v0")

# Reset the environment
env.reset()

# Interact with the environment
for _ in range(10):
    action = env.action_space.sample()  # Sample a random action
    obs, reward, done, info = env.step(action)  # Take a step
    print(f"Observation: {obs}, Reward: {reward}, Done: {done}")

    if done:
        env.reset()
        
```

## Data
Download data at https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2F01JJZY&version=DRAFT#

Extract and put data/ at RL4Grid/RL4Grid/


## Test
```bash
cd RL4Grid/RL4Grid
python test.py
```