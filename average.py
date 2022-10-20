import numpy as np


# qmix1 = np.load('rewards_3uav_qmix_s2.npy')
device2 = np.loadtxt('2022-06-15_17-10-50_Episode3000_size2000_noise0.0_batch64.txt')

avg_2 = np.nanmean(device2)