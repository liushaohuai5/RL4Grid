import numpy as np
import copy

class GridSimWrapper:
    def __init__(self, env):

        self.env = env

        self.last_obs = None
        self.step_cnt = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.step_cnt += 1
        self.last_obs = copy.deepcopy(observation)
        return observation, reward, done, info

    def step_only_attack(self, action_one_hot):
        action = np.where(action_one_hot > 0)[0][0]
        if action == len(action_one_hot) - 1:
            action = None
        observation, reward, done, info = self.env.step_only_attack(action)
        self.last_obs = copy.deepcopy(observation)
        return observation, reward, done, info

    def load_and_step(self, snapshot, action):
        new_snapshot, reward, done, info = self.env.get_results(snapshot, action)
        return new_snapshot, reward, done, info

    def get_snapshot(self):
        return self.env.get_snapshot()

    def reset_snapshot(self):
        observation = self.env.reset()
        sample_idx = self.env.sample_idx
        return (observation, sample_idx)

    def reset(self, ori_obs=False, start_sample_idx=None, seed=0):
        self.step_cnt = 0
        observation = self.env.reset(start_sample_idx=start_sample_idx)
        self.last_obs = copy.deepcopy(observation)
        return observation

    def close(self):
        self.env.close()
