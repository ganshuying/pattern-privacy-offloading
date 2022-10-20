import numpy as np
import queue
from functools import cmp_to_key
import gym
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)  # 设精度为3
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from gym import spaces
class MECEnv(gym.Env):

    def __init__(self, num_device=5,
                 edge_computing_capacity=5000, # 500
                 cloud_computing_capacity=30000, # 4000
                 transmit_rate=0.5,
                 edge_CPU_coefficient=10e-16, # 10e-26
                 cloud_CPU_coefficient=10e-11, # 10e-11
                 transmit_power=0.3,
                 energy_ratio=0.05,
                 wait_ratio=1):
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
        self.queue_max_size = 5
        self.transmit_queue = []
        self.task_feature_dim = 8
        self.info = dict()

        high = np.ones_like(self.reset()) * 100
        self.observation_space = spaces.Box(-high,high,dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self):
        self.current_step = 0
        self.generate_task()
        self.request_queue = []
        self.precess_queue = []
        self.transmit_queue = []
        self.generate_task()
        self.current_task = self.get_next_task()
        self.current_process = [0] * 8
        self.count_wrong = 0
        return self.get_obs()

    def get_obs(self):
        # current_request_feature = np.zeros(self.task_feature_dim)
        # current_process_feature = np.zeros(self.task_feature_dim)
        request_queue_feature = np.zeros(shape=(self.queue_max_size, self.task_feature_dim))
        process_queue_feature = np.zeros(shape=(self.queue_max_size, self.task_feature_dim))
        transmit_queue_feature = np.zeros(shape=(self.queue_max_size, self.task_feature_dim))
        current_request_feature = np.array(self.current_task)
        current_process_feature = np.array(self.current_process)

        # 归一化防止nan
        current_process_feature[1] = current_process_feature[1] / 2000
        current_request_feature[1] = current_request_feature[1] / 2000

        current_request_feature[2] = current_request_feature[2] / 3000
        current_process_feature[2] = current_process_feature[2] / 3000
        for i, t in enumerate(self.request_queue):
            task = t.copy()
            task[1] = task[1] / 2000
            task[2] = task[2] / 3000

            # print(task)
            # print(self.request_queue[i])
            request_queue_feature[i] = task

        for i, t in enumerate(self.precess_queue):
            task = t.copy()
            task[1] = task[1] / 2000
            task[2] = task[2] / 3000
            process_queue_feature[i] = task

        for i, t in enumerate(self.transmit_queue):
            task = t.copy()
            task[1] = task[1] / 2000
            task[2] = task[2] / 3000

            transmit_queue_feature[i] = task

        state = np.concatenate(
            (
                current_request_feature.flatten(),
                current_process_feature.flatten(),
                request_queue_feature.flatten(),
                process_queue_feature.flatten(),
                transmit_queue_feature.flatten(),
            )
        )
        self.info['current_request'] = self.current_task
        self.info['current_process'] = self.current_process
        self.info['request_queue'] = self.request_queue
        self.info['process_queue_feature'] = self.precess_queue
        self.info['transmit_queue_feature'] = self.transmit_queue
        return state

    def step(self, action):
        """

        action:[0,1] 0 本地执行，1远程执行
        """
        if self.current_task != [0] * 8:
            if action == 0:
                """
                加入执行队列
                """
                if (len(self.precess_queue) == self.queue_max_size):
                    self.count_wrong += 1
                else:
                    self.precess_queue.append(self.current_task)
            if action == 1:
                if (len(self.transmit_queue) == self.queue_max_size):
                    self.count_wrong += 1
                else:
                    self.transmit_queue.append(self.current_task)

        reward = 0
        # 更新各个队列
        reward += self.update_process()
        # print("local reward:" , reward)
        reward += self.update_transmit()
        reward = -reward
        # print("transmit reward + local:", reward)
        # 更新队列里的等待时长
        self.update_wait_time()

        self.generate_task()
        self.current_task = self.get_next_task()
        self.current_step += 1
        done = False
        if self.current_step == 100:  # 一个episode的步长
            done = True

        return self.get_obs(), reward, done,{}

    def get_next_task(self):

        # 先高响应比优先排序

        while len(self.request_queue) > 0:
            data_size, cpu_cycles, delay_constraints, aoi, process_time, _, _, _ = self.request_queue[0]
            # todo: 需要去处理
            if aoi > delay_constraints:
                self.request_queue.pop()
                self.count_wrong += 1
            else:
                break
        if (len(self.request_queue)) > 0:

            return self.request_queue.pop(0)

        else:
            return [0] * 8

    def generate_task(self):
        # 任务随机到达
        self.generate_task_prob = np.array([0.1] * self.num_device)
        flag_data_arrival = np.random.rand(self.num_device) <= self.generate_task_prob
        # new_task = generate_task(self.num_device)
        new_task = self.generate_task_info()
        new_task = new_task[flag_data_arrival]
        new_task = new_task.tolist()

        # 插入到队列
        insert_num = self.queue_max_size - len(self.request_queue)
        self.request_queue.extend(new_task[0:insert_num])

    def update_process(self):
        """
        更新执行队列
        """
        reward = 0
        # energy = 0
        ## 判断当前是否有任务正在执行
        if self.current_process[1] != 0:
            self.current_process[1] -= self.edge_computing_capacity
            self.current_process[1] = np.maximum(0, self.current_process[1])
            # 处理时长加1
            self.current_process[4] += 1
            if self.current_process[1] == 0:  # 执行完成
                # energy = self.edge_CPU_coefficient * self.edge_computing_capacity * self.edge_computing_capacity * self.current_process[0]
                reward += (self.current_process[4] + self.current_process[3]) * self.wait_ratio  # 排队时长＋处理时长+能耗
                # 添加本地计算能耗
                reward += (self.current_process[5] * self.energy_ratio)
        else:  #
            # 1.排序 从队列中选择一个任务执行
            if (len(self.precess_queue) != 0):
                # 先根据响应比排序

                # self.precess_queue.sort(key=lambda x:(
                #         (x[3] + x[1] / self.computing_capacity) / (x[1] / self.computing_capacity)
                # ),reverse=True) # 从大到小

                self.current_process = self.precess_queue.pop(0)
                self.current_process[1] -= self.edge_computing_capacity
                self.current_process[1] = np.maximum(0, self.current_process[1])
                # 处理时长加1
                self.current_process[4] += 1
                if self.current_process[1] == 0:  # 执行完成
                    # energy = self.edge_CPU_coefficient * self.edge_computing_capacity * self.edge_computing_capacity * self.current_process[0]
                    reward += (self.current_process[4] + self.current_process[3]) * self.wait_ratio  # 排队时长＋处理时长

                    # 添加本地计算能耗
                    reward += (self.current_process[5] * self.energy_ratio)
            else:
                self.current_process = [0] * 8
        return reward
        # 1.判断是否有正在执行的任务
        # 有：
        # 2。高响应比排序

    def update_transmit(self):
        reward = 0
        if len(self.transmit_queue) != 0:
            temp = np.array(self.transmit_queue)  # [transmit_num,5]
            temp[:, 0] = temp[:, 0] - self.transmit_rate
            temp[:, 0] = np.maximum(temp[:, 0], 0)
            temp[:, 4] += 1  # 传送时长加一

            # 选择传送完成的
            success_trans = temp[temp[:, 0] == 0]  # [success_num,5]

            # 3:排队时长 ， 4：传输时长 6云端计算能耗 , 4*power 传输能耗 7 云端计算时延
            reward = np.sum(
                (success_trans[:, 3] + success_trans[:, 4] + success_trans[:, 7])
                * self.wait_ratio +
                (success_trans[:, 6] + success_trans[:, 4] * self.transmit_power)
                * self.energy_ratio
            )

            remaining = temp[temp[:, 0] > 0]
            remaining = remaining.reshape((-1, self.task_feature_dim))
            self.transmit_queue = remaining.tolist()
        return reward

    def update_wait_time(self):
        for i in range(len(self.request_queue)):
            self.request_queue[i][3] += 1
        for i in range(len(self.precess_queue)):
            self.precess_queue[i][3] += 1
        # request_tmp = np.array(self.request_queue)
        # request_tmp[:,3] +=1
        # self.request_queue = request_tmp.tolist()
        #
        # process_tmp = np.array(self.precess_queue)
        # process_tmp[:,3] += 1
        # self.precess_queue = process_tmp.tolist()

    def generate_task_info(self):
        num_device = self.num_device
        new_task = np.zeros(shape=(num_device, 8))  # data_size,cpu_cycles,delay_constrants
        new_task[:, 0] = np.random.uniform(0.5, 1, size=(num_device))
        new_task[:, 1] = np.random.randint(5000, 20000, size=(num_device))
        new_task[:, 2] = [3000] * num_device  # 容忍时长
        new_task[:, 3] = [0] * num_device  # 排队时长
        new_task[:, 4] = [0] * num_device  # 处理时长

        # 本地计算能耗
        new_task[:, 5] = self.edge_CPU_coefficient * np.power(self.edge_computing_capacity, 2) * new_task[:, 1]
        # 云端计算能耗
        new_task[:, 6] = self.cloud_CPU_coefficient * np.power(self.edge_computing_capacity, 2) * new_task[:, 1]
        # 云端计算时延
        # new_task[:,7] = new_task[:,1] / self.cloud_computing_capacity
        # 向上取整
        new_task[:, 7] = np.ceil(new_task[:, 1] / self.cloud_computing_capacity)
        return new_task

    def get_greedy_action(self):
        if self.current_task != [0] * 8:
            """
            计算排队时长
            """
            # 本地需要等待时长
            local_wait = 0
            precess_queue = np.array(self.precess_queue)
            if precess_queue.shape[0] != 0:
                local_wait += np.sum(precess_queue[:, 1]) / self.edge_computing_capacity
            local_wait += self.current_process[1] / self.edge_computing_capacity

            # 执行本任务时长
            process_time = self.current_task[1] / self.edge_computing_capacity

            all_local_time = local_wait + process_time

            # 远程执行
            transmit_time = self.current_task[0] / self.transmit_rate
            edge_time = self.current_task[1] / self.cloud_computing_capacity

            all_transmit_time = transmit_time + edge_time

            if all_local_time > transmit_time:
                return 1
            else:
                return 0


def generate_task(num_device):
    new_task = np.zeros(shape=(num_device, 5))  # data_size,cpu_cycles,delay_constrants
    new_task[:, 0] = np.random.uniform(0.5, 1, size=(num_device))
    new_task[:, 1] = np.random.randint(5000, 20000, size=(num_device))
    new_task[:, 2] = [3000] * num_device  # 容忍时长
    new_task[:, 3] = [0] * num_device  # 排队时长
    new_task[:, 4] = [0] * num_device  # 处理时长

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

if __name__ == '__main__':
    from itertools import count

    env = MECEnv()
    state = env.reset()

    num_episodes = 10
    episodic_rewards = []
    steps_rewards = []
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
            action = env.get_greedy_action()
            if debug:
                info = env.info
                print("step: {}".format(t).center(60,"-"))
                print("current_request: ", info['current_request'])
                print("current_process: ", info["current_process"])
                print("request_queue: ", info['request_queue'])
                print("process_queue: ", info['process_queue_feature'])
                print("transmit_queue: ", info['transmit_queue_feature'])

            # random
            # action = np.random.randint(0,2)
            # print(action)
            print("action: ", action)
            next_state, reward, done,_ = env.step(action)
            reward = -reward
            steps_rewards.append(reward)

            print("reward: ",reward)
            # import pdb; pdb.set_trace()
            total_reward += float(reward)

            # Move to the next state
            state = next_state

            if done:
                # episode_durations.append(t + 1)
                # plot_durations()
                break
        episodic_rewards.append(total_reward)
        # print(env.count_wrong)
        print(total_reward)
        # 每个episode都重置一次噪声缓冲池，policy和target网络均有噪声池
        if (i_episode + 1) % 10 == 0:
            print("Episode: {}, Score: {}".format(i_episode + 1, np.mean(episodic_rewards[-10:])))

        np.savetxt("greedy.txt",episodic_rewards,fmt="%.2f")
