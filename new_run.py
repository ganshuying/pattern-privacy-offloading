import sys
import gym
import math
import random
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 初始化环境
from common.replay_buffer import ReplayMemory
from env.mec_env_v8 import MECEnv
# from my_env.policy.noisy_dqn import DQN
from policy.net import NoisyNet
from policy.noisy_dqn import DQN

from arguments import get_common_args
args = get_common_args()
print(args)

env = MECEnv(num_device=args.num_device,
             edge_computing_capacity=args.edge_computing_capacity,
             cloud_computing_capacity=args.cloud_computing_capacity,
             transmit_rate=args.transmit_rate,
             edge_CPU_coefficient=args.edge_CPU_coefficient,
             cloud_CPU_coefficient=args.cloud_CPU_coefficient,
             transmit_power=args.transmit_power,
             energy_ratio=args.energy_ratio,
             wait_ratio=args.wait_ratio)
env.reset()
input_dim = len(env.get_obs())
action_dim = 2

# 设置超参数
device = args.device
BATCH_SIZE = args.batch_size
GAMMA = args.gamma
EPS_START = 0.9  # 起始值
EPS_END = 0.05  # 最小值
EPS_DECAY = 200  # 调整eps-greedy的值
TARGET_UPDATE = args.target_update  # 多少步长更新一次

sigma = args.sigma
num_episodes = args.num_episodes  # episode数量设置
minimal_size = args.minimal_size
# 初始化Q网络
# policy_net = NoisyNet(sigma=sigma, input_dim=input_dim, action_dim=action_dim).to(device)
# target_net = NoisyNet(sigma=sigma, input_dim=input_dim, action_dim=action_dim).to(device)
# target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()

# optimizer = optim.RMSprop(policy_net.parameters())  # 参数优化器
memory = ReplayMemory(10000)  # replay buffer
agent = DQN(sigma=sigma,input_dim=input_dim,action_dim=2,device=args.device)


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
        next_state, reward, done= env.step(action.item())
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
    episodic_rewards.append(total_reward)
    # 每个episode都重置一次噪声缓冲池，policy和target网络均有噪声池
    agent.reset_noisy()
    if (i_episode + 1) % 50 == 0:
        print("Episode: {}, Score: {}".format(i_episode + 1, np.mean(episodic_rewards[-10:])))




#file_name = "Episode"+num_episodes+"_"
current_time = "{}_".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
params_str = "Episode{}_size{}_noise{}_batch{}.txt".format(num_episodes,minimal_size,sigma,BATCH_SIZE)
# file_name = current_time.join(params_str)
file_name = current_time+params_str


episodes_list = list(range(len(episodic_rewards)))
with open(file_name, 'a') as fw: # dpql2是sigma为0，episode 100，seed 100.
    fw.write(args.__str__())
    fw.write('\n')
    for rr in episodic_rewards:
        fw.write(str(rr))
        fw.write(' ')
    fw.write('\n')
plt.plot(episodes_list, episodic_rewards)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format('nosiy'))
plt.show()



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
