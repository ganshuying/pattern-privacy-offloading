import random

import numpy as np
import queue
from functools import cmp_to_key
import gym

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)  # 设精度为3
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from gym import spaces

"""
2022-7-30: 传输有5个队列，哪个队列少排到哪个队列
"""




class MECEnv(gym.Env):

    def __init__(self, num_device=5,
                 edge_computing_capacity=500,
                 cloud_computing_capacity=4000,
                 transmit_rate=5,
                 edge_CPU_coefficient=10e-26,
                 cloud_CPU_coefficient=10e-11,
                 transmit_power=0.2,
                 energy_ratio=1,
                 wait_ratio=1,
                 seed=100):
        super(MECEnv, self).__init__()
        self.num_device = num_device
        self.edge_computing_capacity = edge_computing_capacity
        self.cloud_computing_capacity = cloud_computing_capacity
        self.transmit_rate = transmit_rate  # 2Mb/s
        self.edge_CPU_coefficient = edge_CPU_coefficient
        self.cloud_CPU_coefficient = cloud_CPU_coefficient
        self.transmit_power = transmit_power
        self.energy_ratio = energy_ratio
        self.wait_ratio = wait_ratio

        # 记录时间步骤
        self.current_step = 0
        self.count_wrong = 0
        self.request_queue = []
        self.precess_queue = []
        self.queue_max_size = 20
        self.transmit_queue = []
        self.task_feature_dim = 4
        self.info = dict()

        self.num_device = 5
        self.local_computing_waiting_len = 0  # 本地等待计算的长度
        self.local_computing_max_len = 100  # 本地内存最大长度
        self.max_num_transmit = 5  # 最大传输个数
        self.transmit_overflow_penalty = 100  # 传输队列超出惩罚项
        self.transmit_overflow_num = 0  # 记录传输失败个数

        # high = np.ones_like(self.get_obs2()) * 100
        high = np.ones(6 + self.queue_max_size * 4) * 100
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.set_seed(seed)
        self.reset()

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        # self.observation_space =

    def reset(self):
        self.local_computing_waiting_len = 0  # 本地等待计算的长度
        self.transmit_overflow_num = 0  # 记录传输失败个数
        # self.local_transmit_len = 0

        self.local_transmits_len = [0]*self.max_num_transmit  # 记录5个队列各自的剩余传输量
        self.local_transmits_len = np.array(self.local_transmits_len, dtype=np.float32)
        self.task_count = 0  # 记录任务的个数
        self.request_overflow_num = 0  # 记录请求队列溢出的个数
        self.request_queue = []
        self.precess_queue = []
        self.transmit_queue = []
        self.current_step = 0
        self.count_wrong = 0
        self.generate_task()

        # self.generate_task()
        self.current_task = self.get_next_task()

        high = np.ones_like(self.get_obs()) * 100
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        return self.get_obs()

    def get_obs(self):
        request_queue_feature = np.zeros(shape=(self.queue_max_size, self.task_feature_dim), dtype=np.float32)
        local_computing_feature = np.zeros(1, dtype=np.float32)  # 计算队列的计算量
        local_transmit_feature = np.zeros(self.max_num_transmit, dtype=np.float32)  # 传输队列中的任务数量
        current_request_feature = np.zeros(self.task_feature_dim, dtype=np.float32)

        for i in range(len(self.request_queue)):
            data_size, cpu, delay_constrants, wait_time = self.request_queue[i]
            request_queue_feature[i][0] = data_size
            request_queue_feature[i][1] = cpu / 1000
            request_queue_feature[i][2] = delay_constrants / 1000  # 归一化1000
            request_queue_feature[i][3] = wait_time

        local_computing_feature[0] = self.local_computing_waiting_len / self.local_computing_max_len
        # local_transmit_feature[0] = self.local_transmit_len
        for i in range(self.max_num_transmit):
            local_transmit_feature[i] = self.local_transmits_len[i]
        for i in range(self.task_feature_dim):
            current_request_feature[i] = self.current_task[i]
        current_request_feature[1] /= 1000
        current_request_feature[2] /= 1000

        state = np.concatenate(
            (
                request_queue_feature.flatten(),
                local_computing_feature.flatten(),
                local_transmit_feature.flatten(),
                current_request_feature.flatten()
            )
        )
        self.info['request_queue'] = request_queue_feature
        self.info["local_computing_feature"] = local_computing_feature \
                                               * self.local_computing_max_len
        self.info["transmit_feature"] = local_transmit_feature
        self.info["current_request_feature"] = current_request_feature

        return state

    def step(self, action):
        if self.current_task != [0] * self.task_feature_dim:
            self.task_count += 1

        reward = self.get_agent_action(action)

        """
        1. 更新request 队列的等待时长
        2. 更新传输队列
        3. 更新处理队列
        """
        # 更新计算队列
        self.local_computing_waiting_len -= self.edge_computing_capacity
        self.local_computing_waiting_len = np.maximum(self.local_computing_waiting_len, 0)

        # self.update_transmit2()
        # self.local_transmit_len = self.local_transmit_len - self.transmit_rate
        # self.local_transmit_len = np.maximum(self.local_transmit_len, 0)
        # 更新requesting等待时长
        for i in range(len(self.request_queue)):
            self.request_queue[i][3] += 1

        self.updata_transmit()

        self.generate_task()
        self.current_task = self.get_next_task()
        self.current_step += 1
        done = False
        if self.current_step == 100:  # 一个episode的步长
            done = True
        return self.get_obs(), -reward, done, {}
    def updata_transmit(self):
        self.local_transmits_len = self.local_transmits_len - self.transmit_rate
        self.local_transmits_len[self.local_transmits_len < 0] = 0

    def get_agent_action(self, action):
        if self.current_task == [0] * self.task_feature_dim:
            return 0  # 如果没有任务，奖励为0
        if action == 0:
            # if self.local_computing_waiting_len + self.current_task > self.local_computing_max_len:
            # 如果超出本地计算内存限制， 直接抛掉

            reward = 0
            time_cost = 0
            energy_cost = 0
            # 计算队列排队 计算时长
            computing_wait = self.local_computing_waiting_len / self.edge_computing_capacity
            computing_time = self.current_task[1] / self.edge_computing_capacity
            # requesting_wait = self.current_task[3]
            requesting_wait = 0
            time_cost = computing_wait + computing_time + requesting_wait

            # 计算能耗
            energy_cost = self.edge_CPU_coefficient * \
                          np.power(self.edge_computing_capacity, 2) * self.current_task[1]
            reward = self.wait_ratio * time_cost + self.energy_ratio * energy_cost

            # 添加到本地队列中去
            self.precess_queue.append(self.current_task)
            self.local_computing_waiting_len = self.local_computing_waiting_len + self.current_task[1]

            return reward

        else:


            min_value = np.min(self.local_transmits_len)  # 返回传输队列中的最小值和索引
            min_index = np.argmin(self.local_transmits_len)

            min_value = min_value + self.current_task[0]  # 加入到最小的队列中去
            self.local_transmits_len[min_index] = min_value

            reward = 0

            time_cost = 0
            energy_cost = 0

            # 传输时长、请求队列等待时长、云端计算时长
            transmit_time = min_value / self.transmit_rate
            # requesting_wait = self.current_task[3]
            requesting_wait = 0
            cloud_computing_time = self.current_task[1] / self.cloud_computing_capacity

            # 传输能耗，
            transmit_energy = transmit_time * self.transmit_power
            cloud_computing_energy = self.cloud_CPU_coefficient \
                                     * np.power(self.cloud_computing_capacity, 2) \
                                     * self.current_task[1]

            time_cost = transmit_time + requesting_wait + cloud_computing_time
            energy_cost = transmit_energy + cloud_computing_energy

            reward = self.energy_ratio * energy_cost + self.wait_ratio * time_cost

            # 添加到传输队列
            self.transmit_queue.append(self.current_task)
            return reward

    def get_next_task(self):

        # 先高响应比优先排序

        while len(self.request_queue) > 0:
            data_size, cpu_cycles, delay_constraints, aoi = self.request_queue[0]
            # todo: 需要去处理
            if aoi > delay_constraints:
                self.request_queue.pop()
                self.count_wrong += 1
            else:
                break
        if (len(self.request_queue)) > 0:

            return self.request_queue.pop(0)

        else:
            return [0] * self.task_feature_dim

    def generate_task(self):
        # 任务随机到达
        self.generate_task_prob = np.array([0.2] * self.num_device)
        flag_data_arrival = np.random.rand(self.num_device) > self.generate_task_prob
        # print(flag_data_arrival)
        # new_task = generate_task(self.num_device)
        new_task = self.generate_task_info()
        new_task = new_task[flag_data_arrival]
        new_task = new_task.tolist()

        # 插入到队列
        insert_num = self.queue_max_size - len(self.request_queue)

        if insert_num < len(new_task):
            self.request_overflow_num = self.request_overflow_num + (len(new_task) - insert_num)

        insert_num = np.minimum(len(new_task), insert_num)
        self.request_queue.extend(new_task[0:insert_num])

    def generate_task_info(self):
        num_device = self.num_device
        new_task = np.zeros(shape=(num_device, self.task_feature_dim))  # data_size,cpu_cycles,delay_constrants
        # new_task[:, 0] = np.random.uniform(0.1, 20, size=(num_device))
        # new_task[:, 1] = np.random.randint(1, 5000, size=(num_device))
        # new_task[:, 0] = np.random.uniform(0.3, 0.5, size=(num_device))
        # new_task[:, 1] = np.random.randint(900, 1100, size=(num_device))
        new_task[:, 0] = np.random.uniform(5, 50, size=(num_device))
        new_task[:, 1] = np.random.randint(500, 2000, size=(num_device))
        new_task[:, 2] = [3000] * num_device  # 容忍时长
        new_task[:, 3] = [0] * num_device  # 排队时长
        return new_task
    def if_local(self):
        # 本地计算
        reward = 0
        time_cost = 0
        energy_cost = 0
        # 计算队列排队 计算时长
        computing_wait = self.local_computing_waiting_len / self.edge_computing_capacity
        computing_time = self.current_task[1] / self.edge_computing_capacity
        # requesting_wait = self.current_task[3]
        requesting_wait = 0
        time_cost = computing_wait + computing_time + requesting_wait

        # 计算能耗
        energy_cost = self.edge_CPU_coefficient * \
                      np.power(self.edge_computing_capacity, 2) * self.current_task[1]
        reward = self.wait_ratio * time_cost + self.energy_ratio * energy_cost
        return reward
    def if_transmit(self):
        min_value = np.min(self.local_transmits_len)  # 返回传输队列中的最小值和索引
        min_index = np.argmin(self.local_transmits_len)
        min_value = min_value + self.current_task[0]  # 加入到最小的队列中去
        # self.local_transmits_len[min_index] = min_value

        reward = 0

        time_cost = 0
        energy_cost = 0

        # 传输时长、请求队列等待时长、云端计算时长
        transmit_time = min_value / self.transmit_rate
        # requesting_wait = self.current_task[3]
        requesting_wait = 0
        cloud_computing_time = self.current_task[1] / self.cloud_computing_capacity

        # 传输能耗，
        transmit_energy = transmit_time * self.transmit_power
        cloud_computing_energy = self.cloud_CPU_coefficient \
                                 * np.power(self.cloud_computing_capacity, 2) \
                                 * self.current_task[1]

        time_cost = transmit_time + requesting_wait + cloud_computing_time
        energy_cost = transmit_energy + cloud_computing_energy

        reward = self.energy_ratio * energy_cost + self.wait_ratio * time_cost
        return reward

    def get_greedy_action(self):
        if self.current_task == [0] * self.task_feature_dim:
            return 0 # 可以随机反正没意义

        local_reward = self.if_local()
        transmit_reward = self.if_transmit()
        if local_reward > transmit_reward:  # 因为这是正的
            return 1
        else:
            return 0
        # 计算本地计算能耗



def generate_task(num_device):
    new_task = np.zeros(shape=(num_device, 5))  # data_size,cpu_cycles,delay_constrants
    # new_task[:, 0] = np.random.uniform(0.3, 0.5, size=(num_device))
    # new_task[:, 1] = np.random.randint(900, 1100, size=(num_device))
    new_task[:, 0] = np.random.uniform(5, 50, size=(num_device))
    new_task[:, 1] = np.random.randint(500, 2000, size=(num_device))
    new_task[:, 2] = [3000] * num_device  # 容忍时长
    new_task[:, 3] = [0] * num_device  # 排队时长

    return new_task


# class HRRN():
#     def __init__(self , max_size):

# if __name__ == '__main__':
#     env = MECEnv()
#     state = env.reset()
#     print(state.shape)
#
#     # print(state)
#     print(env.info)
#     s, r, done = env.step(0)
#     # print(s.shape)
#     print(env.info)
#     s, r, done = env.step(1)
#     print(env.info)
#     # print(s)

def test_greedy():
    from itertools import count
    seed = np.random.randint(10000)
    env = MECEnv(seed = seed)

    # state = env.reset()

    num_episodes = 100
    episodic_rewards = []
    steps_rewards = []
    task_nums = []
    # debug = Ture
    debug = False
    for i_episode in range(num_episodes):
        # if i_episode % 10 == 0:
        # print(i_episode)
        # Initialize the environment and state
        state = env.reset()
        total_reward = 0
        for t in count():
            # Select and perform an action
            # greedy
            # action = env.get_greedy_action()
            if debug:
                info = env.info
                print("step: {}".format(t).center(60, "-"))
                print("current_request")
                print(env.current_task)
                for k, v in info.items():
                    print(k)
                    print(v)
                print("传输队列信息")
                print(env.transmit_queue)
            # random
            action = env.get_greedy_action()
            # print(action)
            print("action: ", action)
            next_state, reward, done, _ = env.step(action)
            reward = reward
            steps_rewards.append(reward)
            # print(len(state))
            print("reward: ", reward)
            # import pdb; pdb.set_trace()

            total_reward += float(reward)
            # Move to the next state
            state = next_state

            if done:
                # episode_durations.append(t + 1)
                # plot_durations()
                break

        # print(env.request_overflow_num)
        total_reward = total_reward / env.task_count
        episodic_rewards.append(total_reward)
        task_nums.append(env.task_count)
        # print(env.count_wrong)
        print(total_reward)
        # 每个episode都重置一次噪声缓冲池，policy和target网络均有噪声池
        if (i_episode + 1) % 10 == 0:
            print("Episode: {}, Score: {}".format(i_episode + 1, np.mean(episodic_rewards[-10:])))
            print("mean", np.mean(episodic_rewards))
            print("count mean -10,", np.mean(task_nums[-10:]), "total mean:", np.mean(task_nums))
            np.savetxt("greedy6.txt", episodic_rewards, fmt="%.2f")

import datetime
current_time = "{}_greedy".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
params_str = ".txt"
file_name = current_time + params_str


def test_random():
    from itertools import count

    env = MECEnv()
    # state = env.reset()

    num_episodes = 1
    episodic_rewards = []
    steps_rewards = []
    task_nums = []
    debug = True
    for i_episode in range(num_episodes):
        # if i_episode % 10 == 0:
        # print(i_episode)
        # Initialize the environment and state
        state = env.reset()
        total_reward = 0
        for t in count():
            # Select and perform an action
            # greedy
            # action = env.get_greedy_action()
            if debug:
                info = env.info
                print("step: {}".format(t).center(60, "-"))
                print("current_request")
                print(env.current_task)
                for k, v in info.items():
                    print(k)
                    print(v)
                print("传输队列信息")
                print(env.transmit_queue)
            # random
            action = np.random.randint(2)
            # print(action)
            print("action: ", action)
            next_state, reward, done, _ = env.step(action)
            reward = reward
            steps_rewards.append(reward)
            # print(len(state))
            print("reward: ", reward)
            # import pdb; pdb.set_trace()

            total_reward += float(reward)
            # Move to the next state
            state = next_state

            if done:
                # episode_durations.append(t + 1)
                # plot_durations()
                break

        # print(env.request_overflow_num)
        total_reward = total_reward / env.task_count
        episodic_rewards.append(total_reward)
        task_nums.append(env.task_count)
        # print(env.count_wrong)
        print(total_reward)
        # 每个episode都重置一次噪声缓冲池，policy和target网络均有噪声池
        if (i_episode + 1) % 10 == 0:
            print("Episode: {}, Score: {}".format(i_episode + 1, np.mean(episodic_rewards[-10:])))
            print("mean", np.mean(episodic_rewards))
            print("count mean -10,", np.mean(task_nums[-10:]), "total mean:", np.mean(task_nums))
if __name__ == '__main__':
    test_greedy()
    # test_random()