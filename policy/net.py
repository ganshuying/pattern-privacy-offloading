import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from common.noise_buffer import noisebuffer


class NoisyNet(nn.Module):
    def __init__(self, sigma=0.4, hidden=128, input_dim=128, action_dim=2):
        super(NoisyNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 2)
        self.sigma = sigma
        self.nb = noisebuffer(2, sigma)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        x = self.head(x)
        if self.sigma > 0:
            # x: [batch_size, action_dim]
            x = x.reshape(-1, 2)
            # qmean = [batch_size]
            qmean = torch.mean(x, dim=1)

            eps = [self.nb.sample(float(q)) for q in qmean]  # //
            eps = torch.Tensor(eps)
            return x + eps  # 原始数据加上噪声， x是网络的真是输出，在这加了噪声
        else:
            return x
