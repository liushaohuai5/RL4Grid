import numpy as np

class Disconnect(object):
    def __init__(self, np_random, ppc):
        self.np_random = np_random
        self.lines = [i for i in range(ppc['num_line'])]

        self.prob_dis = ppc['prob_disconnection']
        self.white_list = ppc['white_list_random_disconnection']
        self.hard_bound = ppc['hard_overflow_bound']
        self.soft_bound = ppc['soft_overflow_bound']
        self.num_line = ppc['num_line']
        self.max_steps_to_reconnect_line = ppc['max_steps_to_reconnect_line']
        self.max_steps_soft_overflow = ppc['max_steps_soft_overflow']
        self.steps_to_reconnect_line = [0 for _ in range(ppc['branch'].shape[0])]
        self.count_soft_overflow_steps = [0 for _ in range(ppc['branch'].shape[0])]

    # Randomly cut one line from white list
    def random_cut(self, attack_all=False):
        if self.np_random.rand() < self.prob_dis:
            if not attack_all:
                dis_line_id = self.np_random.choice(self.white_list)
            else:
                dis_line_id = self.np_random.choice(self.lines)
            return [dis_line_id]
        return []

    # Find lines meetings soft & hard overflow
    def overflow(self, rho):
        hard_overflow_ids = np.where(rho > self.hard_bound)[0] 
        soft_overflow_ids = np.intersect1d(np.where(rho > self.soft_bound),
                                       np.where(rho <= self.hard_bound))
        return hard_overflow_ids, soft_overflow_ids

    # Count soft overflow steps
    def count_soft_steps(self, soft_overflow_ids):
        dis_ids = []
        for i in range(self.num_line):
            if i in soft_overflow_ids:
                self.count_soft_overflow_steps[i] += 1
                if self.count_soft_overflow_steps[i] >= self.max_steps_soft_overflow:
                    dis_ids.append(i)
            else:
                self.count_soft_overflow_steps[i] = 0
        return dis_ids

    # If the line is to be & can be cut: cut
    # If the line has been cut before: step -= 1
    def update_reconnect_steps(self, cut_line_ids):
        for i in range(self.num_line):
            if i in cut_line_ids:
                if self.steps_to_reconnect_line[i] == 0:
                    self.steps_to_reconnect_line[i] = self.max_steps_to_reconnect_line
            else:
                if self.steps_to_reconnect_line[i] <= 0:
                    self.steps_to_reconnect_line[i] = 0
                else:
                    self.steps_to_reconnect_line[i] -= 1

    def get_disc_line(self, rho, attack_all=False, attack=None, two_player=False):
        rho = np.array(rho)

        # if is_test:
        if not two_player:
            dis_line_id = self.random_cut(attack_all=attack_all)
        else:
            if not attack_all:
                dis_line_id = [self.white_list[attack]] if attack else []
            else:
                dis_line_id = [self.lines[attack]] if attack else []
        hard_overflow_ids, soft_overflow_ids = self.overflow(rho)
        dis_softoverflow_ids = self.count_soft_steps(soft_overflow_ids)

        cut_line_ids = dis_line_id + dis_softoverflow_ids + hard_overflow_ids.tolist()
        cut_line_ids = list(set(cut_line_ids))

        self.update_reconnect_steps(cut_line_ids)
        final_cut_line_ids = np.where(np.asarray(self.steps_to_reconnect_line) > 0)

        return [idx for idx in final_cut_line_ids[0]]

    def load_from_obs(self, obs):
        self.steps_to_reconnect_line = obs.steps_to_reconnect_line
        self.count_soft_overflow_steps = obs.count_soft_overflow_steps

