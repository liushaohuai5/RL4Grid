# RL4Grid Environment

A fully open-sourced reinforcement learning environment, designed for power system optimal dispatch problem. 
Underlying power flow simulation is based on [PyPower](https://github.com/rwl/PYPOWER/tree/master).

## Features

- General actions: continuous generator active power adjustments
- Adversarial actions: discrete transmission line attack choices (Not available yet)
- Networks: IEEE14, 39, 57, 300 systems and a 126 system (modified from IEEE118)

## Installation

### Install via `pip`

You can install the environment package using `pip`:

```bash
pip install git+https://github.com/liushaohuai5/RL4Grid.git
```

## Usage Example
Once installed, you can use the reinforcement learning environment as follows:

```python

import gym
import RL4Grid  # Import your environment

# Create the environment
env = RL4Grid.make_gridsim('IEEE14')

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

## Modifying Pypower
Copy casefiles to pypower path
```bash
cp model_jm/case* [pypower_package_path]
```

## Data
Download data at https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2F01JJZY&version=DRAFT#

Extract and put data/ at the python package location RL4Grid/data/


## Test
```bash
cd RL4Grid/
python test.py
```