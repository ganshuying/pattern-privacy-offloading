import numpy as np
import queue
from functools import cmp_to_key
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)  # 设精度为3
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class MECEnv:
    def _init__(self):
        self.num_device = 5
        self.edge_computing_capacity = 500
        self.cloud_computing_capacity = 2000
        self.transmit_rate = 0.2  # 2Mb/s
        self.edge_CPU_coefficient = 10^-26
        self.cloud_CPU_coefficient = 10^-11

        # 记录时间步骤
        self.current_step = 0
        self.count_wrong = 0
        self.request_queue = []
        self.precess_queue = []
        self.queue_max_size = 5
        self.transmit_queue = []
        self.task_feature_dim = 5
        self.info=dict()
    def reset(self):
        self.current_step = 1
        self.generate_task()
        self.request_queue = []
        self.precess_queue = []
        self.transmit_queue = []
        self.current_task = self.get_next_task()
        self.current_process = [0, 0, 0, 0, 0]
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
        for i, task in enumerate(self.request_queue):
            task[1] = task[1] / 2000
            task[2] = task[2] / 3000
            request_queue_feature[i] = task

        for i, task in enumerate(self.precess_queue):
            task[1] = task[1] / 2000
            task[2] = task[2] / 3000
            process_queue_feature[i] = task

        for i, task in enumerate(self.transmit_queue):
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
        self.info['current_request'] = current_request_feature
        self.info['current_process'] = current_process_feature
        self.info['request_queue'] = request_queue_feature
        self.info['process_queue_feature'] = process_queue_feature
        self.info['transmit_queue_feature'] = transmit_queue_feature
        return state

    def step(self, action):
        """

        action:[0,1] 0 本地执行，1远程执行
        """
        if self.current_task != [0, 0, 0, 0, 0]:
            if action == 0:
                """
                加入执行队列
                """
                if (len(self.precess_queue) == self.queue_max_size):
                    self.count_wrong += 1
                else:
                    self.precess_queue.append(self.current_task)
            if action == 1:
                if(len(self.transmit_queue) ==self.queue_max_size):
                    self.count_wrong+=1
                else:
                    self.transmit_queue.append(self.current_task)

        reward = 0
        # 更新各个队列
        reward += self.update_process()
        reward += self.update_transmit()

        # 更新队列里的等待时长
        self.update_wait_time()

        self.generate_task()
        self.current_task = self.get_next_task()
        self.current_step += 1
        done = False
        if self.current_step == 100:
            done = True

        return self.get_obs(), reward, done

    def get_next_task(self):

        # 先高响应比优先排序

        while len(self.request_queue) > 0:
            data_size, cpu_cycles, delay_constraints, aoi, process_time = self.request_queue[0]
            # todo: 需要去处理
            process_time = 1
            if aoi > delay_constraints:
                self.request_queue.pop()
                self.count_wrong += 1
            else:
                break
        if (len(self.request_queue)) > 0:

            return self.request_queue.pop(0)

        else:
            return [0, 0, 0, 0, 0]

    def generate_task(self):
        # 任务随机到达
        self.generate_task_prob = np.array([0.1] * self.num_device)
        flag_data_arrival = np.random.rand(self.num_device) > self.generate_task_prob
        new_task = generate_task(self.num_device)
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
        #energy = 0
        ## 判断当前是否有任务正在执行
        if self.current_process[1] != 0:
            self.current_process[1] -= self.edge_computing_capacity
            self.current_process[1] = np.maximum(0, self.current_process[1])
            # 处理时长加1
            self.current_process[4] += 1
            if self.current_process[1] == 0: # 执行完成
                #energy = self.edge_CPU_coefficient * self.edge_computing_capacity * self.edge_computing_capacity * self.current_process[0]
                reward += self.current_process[4] + self.current_process[3] # 排队时长＋处理时长+能耗

        else:#
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
                    #energy = self.edge_CPU_coefficient * self.edge_computing_capacity * self.edge_computing_capacity * self.current_process[0]
                    reward += self.current_process[4] + self.current_process[3]  # 排队时长＋处理时长
            else:
                self.current_process = [0, 0, 0, 0, 0]
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
            temp[:, 4] += 1                         # 传送时长加一

            # 选择传送完成的
            success_trans = temp[temp[:, 0] == 0]  # [success_num,5]
            reward = np.sum(success_trans[:, 3] + success_trans[:, 4])

            remaining = temp[temp[:, 0] > 0]
            remaining = remaining.reshape((-1, 5))
            self.transmit_queue = remaining.tolist()
        return reward

    def update_wait_time(self):
        for i in range(len(self.request_queue)):
            self.request_queue[i][3]+=1
        for i in range(len(self.precess_queue)):
            self.precess_queue[i][3] += 1
        # request_tmp = np.array(self.request_queue)
        # request_tmp[:,3] +=1
        # self.request_queue = request_tmp.tolist()
        #
        # process_tmp = np.array(self.precess_queue)
        # process_tmp[:,3] += 1
        # self.precess_queue = process_tmp.tolist()




def generate_task(num_device):
    new_task = np.zeros(shape=(num_device, 5))  # data_size,cpu_cycles,delay_constrants
    new_task[:, 0] = np.random.uniform(0.3, 0.5, size=(num_device))
    new_task[:, 1] = np.random.randint(900, 1100, size=(num_device))
    new_task[:, 2] = [3000] * num_device # 容忍时长
    new_task[:, 3] = [0] * num_device  # 排队时长
    new_task[:, 4] = [0] * num_device  # 处理时长
    return new_task


# class HRRN():
#     def __init__(self , max_size):

if __name__ == '__main__':
    env = MECEnv()
    state = env.reset()
    print(state.shape)

    # print(state)
    print(env.info)
    s, r, done = env.step(0)
    # print(s.shape)
    print(env.info)
    s, r, done = env.step(1)
    print(env.info)
    # print(s)
