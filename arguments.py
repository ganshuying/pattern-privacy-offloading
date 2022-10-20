import argparse

"""
Here are the param for the training

"""


def get_common_args():
    parser = argparse.ArgumentParser()
    # the training setting
    parser.add_argument('--batch_size', type=int, default=128, help='the difficulty of the game')
    parser.add_argument('--gamma', type=float, default=0.98, help='the version of the game')
    parser.add_argument('--target_update', type=int, default=10, help='the map of the game')
    parser.add_argument('--num_episodes', type=int, default=1500, help='random seed')
    parser.add_argument('--minimal_size', type=int, default=1000, help='how many steps to make an action')
    parser.add_argument('--device', type=str, default='cpu', help='how many steps to make an action')
    parser.add_argument('--sigma', type=float, default=0.0, help='how many steps to make an action')

    # the environment setting
    parser.add_argument('--num_device', type=int, default=5, help='the algorithm to train the agent')
    parser.add_argument('--edge_computing_capacity', type=int, default=500, help='total time steps')
    parser.add_argument('--cloud_computing_capacity', type=int, default=4000, help='the number of episodes before once training')
    parser.add_argument('--transmit_rate', type=float, default=0.2, help='whether to use the last action to choose action')
    parser.add_argument('--edge_CPU_coefficient', type=float, default=10e-26, help='whether to use one network for all agents')
    parser.add_argument('--cloud_CPU_coefficient', type=float, default=10e-11, help='discount factor')
    parser.add_argument('--transmit_power', type=int, default=23, help='optimizer')
    parser.add_argument('--energy_ratio', type=float, default=0.0, help='how often to evaluate the model')
    parser.add_argument('--wait_ratio', type=float, default=1.0, help='number of the epoch to evaluate the agent')


    args = parser.parse_known_args()[0]
    return args