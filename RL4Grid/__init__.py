from .base_env import Environment
from .env_wrapper import GridSimWrapper

def make_gridsim(network, is_test=False, two_player=False, attack_all=False):
    env = Environment(network, "EPRIReward", is_test=is_test, two_player=two_player, attack_all=attack_all)
    return GridSimWrapper(env)