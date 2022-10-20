import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from common.noise_buffer import noisebuffer
from common.replay_buffer import Transition
from policy.net import NoisyNet
import random
import numpy as np

class DQN(nn.Module):
    def __init__(self, sigma=0.0, hidden_dim=128,input_dim=128, action_dim=2,device="cpu",
                 learning_rate=2e-3,
                 gamma=0.98,
                 target_update=10,
                 epsilon_start=0.9,
                 epsilon_end=0.05):
        super(DQN, self).__init__()
        self.action_dim = action_dim
        self.input_dim = input_dim
        self.q_net = NoisyNet(sigma, hidden_dim, input_dim,action_dim).to(device)  # Q网络
        self.target_q_net = NoisyNet(sigma, hidden_dim, input_dim,action_dim).to(device)  # 目标网络
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        # self.epsilon = epsilon  # epsilon-greedy
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器，记录更新次数


        self.device = device
        self.epsilon_start= epsilon_start  # 起始值
        self.epsilon_end = epsilon_end  # 最小值
        self.epsilon = 200  # 调整eps-greedy的值

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def select_action(self,state):
        sample = random.random()
        # eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
        #                 math.exp(-1. * self.count / EPS_DECAY)
        # self.count += 1
        eps_threshold = 0.02
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest value for column of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.q_net(state).max(1)[1].view(1, 1)
        else:
            # action = np.random.randint(self.action_dim)
            return torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)

    def optimize_model(self , transitions):

        # 抽取 batch_size 条数据
        # transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        # import pdb; pdb.set_trace()
        reward_batch = torch.cat(batch.reward)
        BATCH_SIZE = state_batch.shape[0]
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # 先经过网络计算出 Q 然后加 noisy， 他在网络里集成了
        state_action_values = self.q_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_q_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.q_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1
    def reset_noisy(self):
        self.target_q_net.nb.reset()
        self.q_net.nb.reset()