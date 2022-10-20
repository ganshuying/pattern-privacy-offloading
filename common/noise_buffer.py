import bisect
import numpy as np


def kk(x, y):
    return np.exp(-abs(x - y))


def rho(x, y):
    return np.exp(abs(x - y)) - np.exp(-abs(x - y))


class noisebuffer:  # 噪声池
    def __init__(self, m, sigma):
        self.buffer = []
        self.base = {}
        self.m = m
        self.sigma = sigma

    def sample(self, s):  # 采样状态S的噪声值 这块是状态形成噪声参数值的函数 only state no action
        # 有个问题，如果state没有见过怎么半，这是再生核函数，可以在空间中找到关系，它是这么说的，
        buffer = self.buffer
        sigma = self.sigma

        if len(buffer) == 0:
            v0 = np.random.normal(0, sigma)
            v1 = np.random.normal(0, sigma)
            self.buffer.append((s, v0, v1))
            return (v0, v1)  # v0和v1分别代表什么意思？为什么要两个v，分别表示什么意思？？？
        else:
            idx = bisect.bisect(buffer, (s, 0, 0))  # 用于有序序列的查找。
            if len(buffer) == 1:
                if buffer[0][0] == s:
                    return (buffer[0][1], buffer[0][2])
            else:  # 好像在这处理
                if (idx <= len(buffer) - 1) and (buffer[idx][0] == s):
                    return (buffer[idx][1], buffer[idx][2])
                elif (idx >= 1) and (buffer[idx - 1][0] == s):
                    return (buffer[idx - 1][1], buffer[idx - 1][2])
                elif (idx <= len(buffer) - 2) and (buffer[idx + 1][0] == s):
                    return (buffer[idx + 1][1], buffer[idx + 1][2])

        if s < buffer[0][0]:
            mean0 = kk(s, buffer[0][0]) * buffer[0][1]
            mean1 = kk(s, buffer[0][0]) * buffer[0][2]
            var0 = 1 - kk(s, buffer[0][0]) ** 2
            var1 = 1 - kk(s, buffer[0][0]) ** 2
            v0 = np.random.normal(mean0, np.sqrt(var0) * sigma)
            v1 = np.random.normal(mean1, np.sqrt(var1) * sigma)
            self.buffer.insert(0, (s, v0, v1))
        elif s > buffer[-1][0]:
            mean0 = kk(s, buffer[-1][0]) * buffer[0][1]
            mean1 = kk(s, buffer[-1][0]) * buffer[0][2]
            var0 = 1 - kk(s, buffer[-1][0]) ** 2
            var1 = var0
            v0 = np.random.normal(mean0, np.sqrt(var0) * sigma)
            v1 = np.random.normal(mean1, np.sqrt(var1) * sigma)
            self.buffer.insert(len(buffer), (s, v0, v1))
        else:
            idx = bisect.bisect(buffer, (s, None, None))
            sminus, eminus0, eminus1 = buffer[idx - 1]
            # if idx >= len(buffer):
             #   idx = idx - 1
            splus, eplus0, eplus1 = buffer[idx]
            mean0 = (rho(splus, s) * eminus0 + rho(sminus, s) * eplus0) / rho(sminus, splus)  # 用s-计算的噪声函数
            mean1 = (rho(splus, s) * eminus1 + rho(sminus, s) * eplus1) / rho(sminus, splus)  # 用s+计算的噪声函数
            var0 = 1 - (kk(sminus, s) * rho(splus, s) + kk(splus, s) * rho(sminus, s)) / rho(sminus, splus)
            var1 = var0
            v0 = np.random.normal(mean0, np.sqrt(var0) * sigma)  # 为啥这里采样采两个噪声，v0和v1分别代表谁的噪声？
            v1 = np.random.normal(mean1, np.sqrt(var1) * sigma)  # mean1和mean0的区别在哪里？
            self.buffer.insert(idx, (s, v0, v1))
        return (v0, v1)

    def reset(self):
        self.buffer = []