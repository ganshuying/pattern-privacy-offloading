import sys
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 初始化环境
from common.replay_buffer import ReplayMemory
from env.mec_env_v3 import MECEnv
# from my_env.policy.noisy_dqn import DQN
from policy.net import NoisyNet
from policy.noisy_dqn import DQN

from arguments import get_common_args

args = get_common_args()




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 设置超参数
seed = 20013

# env = MECEnv(seed = seed)
env = MECEnv()

env.reset()
input_dim = len(env.get_obs())
action_dim = 2


set_seed(seed)
device = "cpu"
BATCH_SIZE = 32
GAMMA = 0.999

EPS_START = 0.9  # 起始值
EPS_END = 0.05  # 最小值
EPS_DECAY = 200  # 调整eps-greedy的值
TARGET_UPDATE = 10  # 多少步长更新一次

sigma = 0.1
num_episodes = 600  # episode数量设置
minimal_size = 1000
# 初始化Q网络
# policy_net = NoisyNet(sigma=sigma, input_dim=input_dim, action_dim=action_dim).to(device)
# target_net = NoisyNet(sigma=sigma, input_dim=input_dim, action_dim=action_dim).to(device)
# target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()

# optimizer = optim.RMSprop(policy_net.parameters())  # 参数优化器
memory = ReplayMemory(10000)  # replay buffer
agent = DQN(sigma=sigma, input_dim=input_dim, action_dim=2)

episodic_rewards = []
steps_rewards = []
for i_episode in range(num_episodes):
    # if i_episode % 10 == 0:
    # print(i_episode)
    # Initialize the environment and state
    state = torch.Tensor(env.reset()).unsqueeze(0)
    total_reward = 0
    for t in count():
        # Select and perform an action
        action = agent.select_action(state)
        next_state, reward, done = env.step(action.item())
        reward = -reward
        steps_rewards.append(reward)
        reward = torch.tensor([reward], device=device)

        if not done:
            next_state = torch.Tensor(next_state).unsqueeze(0)
        else:
            next_state = None

        # Store the transition in memory
        # no noisy push into buffer
        memory.push(state, action, next_state, reward)
        # import pdb; pdb.set_trace()
        total_reward += float(reward.squeeze(0).data)

        # Move to the next state
        state = next_state

        if len(memory) > minimal_size:
            transitions = memory.sample(BATCH_SIZE)
            agent.optimize_model(transitions)
        if done:
            # episode_durations.append(t + 1)
            # plot_durations()
            break
    # total_reward = total_reward / env.task_count
    episodic_rewards.append(total_reward)
    # 每个episode都重置一次噪声缓冲池，policy和target网络均有噪声池
    agent.reset_noisy()
    if (i_episode + 1) % 50 == 0:
        print("Episode: {}, Score: {}, ep_reward".format(i_episode + 1, np.mean(episodic_rewards[-10:])))
        print("total", total_reward)
        # print("task_count", env.task_count)

episodic_rewards.remove(min(episodic_rewards))
episodes_list = list(range(len(episodic_rewards)))
plt.plot(episodes_list, episodic_rewards)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format('Offloading'))
plt.show()

import datetime

# file_name = "Episode"+num_episodes+"_"
current_time = "{}_".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
params_str = "Episode{}_size{}_noise{}_batch{}.txt".format(num_episodes, minimal_size, sigma, BATCH_SIZE)
# file_name = current_time.join(params_str)
file_name = current_time + params_str
with open(file_name, 'a') as fw:  # dpql2是sigma为0，episode 100，seed 100.
    for rr in episodic_rewards:
        fw.write(str(rr))
        fw.write(' ')
    fw.write('\n')
