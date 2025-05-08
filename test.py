import RL4Grid
from RL4Grid.utilize.form_action import form_action
import numpy as np

env = RL4Grid.make_gridsim(network='IEEE14')
obs = env.reset()
print(obs)
num_gen = env.env.ppc['gen'].shape[0]
adjust_gen_p = np.zeros(num_gen)
adjust_gen_v = np.zeros(num_gen)
obs, reward, done, info = env.step(form_action(adjust_gen_p, adjust_gen_v))
import ipdb
ipdb.set_trace()