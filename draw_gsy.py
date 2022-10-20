import matplotlib.pyplot as plt
import matplotlib
from pylab import *
from scipy.interpolate import make_interp_spline

import numpy as np

# def moving_average(a, window_size):
#     cumulative_sum = np.cumsum(np.insert(a, 0, 0))
#     middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
#     r = np.arange(1, window_size - 1, 2)
#     begin = np.cumsum(a[:window_size - 1])[::2] / r
#     end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
#     return np.concatenate((begin, middle, end))

##读取.npy文件的数据
# np.save('load_data', write_data)

# qmix1 = np.load('rewards_3uav_qmix_s2.npy')
nosie01 = np.loadtxt('Episode3000_size2000_noise0.0_batch128.txt')
#nosie02 = np.loadtxt('Episode2000_size2000_noise0.1_batch128.txt')

#qmix2 = np.loadtxt('rewards_3uav_0.8_qmix_s1.txt')
#qmix3 = np.loadtxt('rewards_3uav_0.8_qmix_s2.txt')

#return_ph = moving_average(nosie2, 39)

#iql1 = np.loadtxt('rewards_3uav_0.8_iql_s1.txt')
#iql2 = np.loadtxt('rewards_3uav_0.8_iql_s2.txt')
# iql3 = np.load('episode_rewards_iql3.npy')
noise11 = np.loadtxt('Episode3000_size2000_noise0.1_batch128.txt')
noise21 = np.loadtxt('Episode3000_size2000_noise0.2_batch128.txt')
noise31 = np.loadtxt('Episode3000_size2000_noise0.3_batch128.txt')


#vdn1 = np.load('rewards_3uav_0.8_vdn_s1.npy')
#vdn2 = np.load('rewards_3uav_0.8_vdn_s2.npy')
# vdn3 = np.load('rewards_3uav_vdn_s3.npy')

# ## 读取.txt文件的数据
# read_data2 = np.loadtxt('test11.txt', encoding='bytes')
# print(read_data2)


## 设置全局字体
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Times New Roman'],
    # 'font.sans-serif': ['SimHei'],
    # 'axes.unicode_minus': False,
    # 'font.size': 100
    # 'font.weight': 'blod',
    "mathtext.fontset": 'stix',


})

## x轴y轴设定（维度相同）
# x = np.arange(points)
# y_mean = np.random.rand(points)
# y1 = np.random.rand(points)

#####################################修改#############
## 截取数据长度
FINAL_STEP = 3000
## 间隔几个点
INTERNAL = 5
#####################################修改#############


## 截取部分数据，并显示在图中（纵坐标）
# qmix11 = qmix1[:FINAL_STEP]
#qmix22 = qmix2[:FINAL_STEP]
#qmix33 = qmix3[:FINAL_STEP]
noise01_z1 = nosie01[:FINAL_STEP]


#iql11 = iql1[:FINAL_STEP]
#iql22 = iql2[:FINAL_STEP]
# iql33 = iql3[:FINAL_STEP]

noise11_z1 = noise11[:FINAL_STEP]
noise21_z1 = noise21[:FINAL_STEP]
noise31_z1 = noise31[:FINAL_STEP]

#vdn11 = vdn1[:FINAL_STEP]
#vdn22 = vdn2[:FINAL_STEP]
# vdn33 = vdn3[:FINAL_STEP]


## 几条曲线求平均,最大最小值
# mean_qmix = np.mean(np.array([qmix11, qmix22, qmix33]), axis=0)
# min_qmix = np.min(np.array([qmix11, qmix22, qmix33]), axis=0)
# max_qmix = np.max(np.array([qmix11, qmix22, qmix33]), axis=0)


#mean_qmix = np.mean(np.array([qmix22, qmix33]), axis=0)
#min_qmix = np.min(np.array([qmix22, qmix33]), axis=0)
#max_qmix = np.max(np.array([qmix22, qmix33]), axis=0)

# mean_qmix = np.mean(np.array([qmix33]), axis=0)
# min_qmix = np.min(np.array([qmix33]), axis=0)
# max_qmix = np.max(np.array([qmix33]), axis=0)


## 隔段取数
#Mean_qmix = mean_qmix[0::INTERNAL]  ## 每隔5个点取一个数（更新x轴长度）
#Min_qmix = min_qmix[0::INTERNAL]
#Max_qmix = max_qmix[0::INTERNAL]



#result_qmix_mean = [i*10 for i in Mean_qmix]
#result_qmix_min = [i*10 for i in Min_qmix]
#result_qmix_max = [i*10 for i in Max_qmix]

result_noise01 = [i*5 for i in noise01_z1]
result_noise11 = [i*5 for i in noise11_z1]
result_noise21 = [i*5 for i in noise21_z1]
result_noise31 = [i*5 for i in noise31_z1]


# mean_iql = np.mean(np.array([iql11, iql22, iql33]), axis=0)
# min_iql = np.min(np.array([iql11, iql22, iql33]), axis=0)
# max_iql = np.max(np.array([iql11, iql22, iql33]), axis=0)
#mean_iql = np.mean(np.array([iql11, iql22]), axis=0)
#min_iql = np.min(np.array([iql11, iql22]), axis=0)
#max_iql = np.max(np.array([iql11, iql22]), axis=0)

#Mean_iql = mean_iql[0::INTERNAL]
#Min_iql = min_iql[0::INTERNAL]
#Max_iql = max_iql[0::INTERNAL]

#result_iql_mean = [i*10 for i in Mean_iql]
#result_iql_min = [i*10 for i in Min_iql]
#result_iql_max = [i*10 for i in Max_iql]

#############

# mean_vdn = np.mean(np.array([vdn11, vdn22, vdn33]), axis=0)
# min_vdn = np.min(np.array([vdn11, vdn22, vdn33]), axis=0)
# max_vdn = np.max(np.array([vdn11, vdn22, vdn33]), axis=0)

#mean_vdn = np.mean(np.array([vdn11, vdn22]), axis=0)
#min_vdn = np.min(np.array([vdn11, vdn22]), axis=0)
#max_vdn = np.max(np.array([vdn11, vdn22]), axis=0)

#Mean_vdn = mean_vdn[0::INTERNAL]
#Min_vdn = min_vdn[0::INTERNAL]
#Max_vdn = max_vdn[0::INTERNAL]


#result_vdn_mean = [i*10 for i in Mean_vdn]
#result_vdn_min = [i*10 for i in Min_vdn]
#result_vdn_max = [i*10 for i in Max_vdn]

#############



## 横轴长度
x = np.arange(FINAL_STEP/INTERNAL)
x = x*INTERNAL


#############******************************平滑操作##############
#model_mean_qmix = make_interp_spline(x, result_qmix_mean)
#model_min_qmix = make_interp_spline(x, result_qmix_min)
#model_max_qmix = make_interp_spline(x, result_qmix_max)

xs = np.linspace(0, FINAL_STEP, 50)    ## 50表示平滑程度

model_noise01 = make_interp_spline(x, result_noise01)
model_noise11 = make_interp_spline(x, result_noise11)
model_noise21 = make_interp_spline(x, result_noise21)
model_noise31 = make_interp_spline(x, result_noise31)

y_noise01 = model_noise01(xs)
y_noise11 = model_noise11(xs)
y_noise21 = model_noise21(xs)
y_noise31 = model_noise31(xs)


#y_mean_qmix = model_mean_qmix(xs)
#y_min_qmix = model_min_qmix(xs)
#y_max_qmix = model_max_qmix(xs)


#model_mean_iql = make_interp_spline(x, result_iql_mean)
#model_min_iql = make_interp_spline(x, result_iql_min)
#model_max_iql = make_interp_spline(x, result_iql_max)


#y_mean_iql = model_mean_iql(xs)
#y_min_iql = model_min_iql(xs)
#y_max_iql = model_max_iql(xs)



#model_mean_vdn = make_interp_spline(x, result_vdn_mean)
#model_min_vdn = make_interp_spline(x, result_vdn_min)
#model_max_vdn = make_interp_spline(x, result_vdn_max)


#y_mean_vdn = model_mean_vdn(xs)
#y_min_vdn = model_min_vdn(xs)
#y_max_vdn = model_max_vdn(xs)

###########****************************************



## 画图（面板）

fig, ax = plt.subplots(1, 1, figsize=(10, 5)) #4,3

# 字体
legend_front = {'family': 'Times New Roman',
# 'weight' : 'normal',
'size'   : 18,
}

xy_descri = {
# 'family': 'SimHei',
'family': 'Times New Roman',
# 'weight' : 'bold',
# 'weight' : 'normal',
'size'   : 18,
}

## 线形，颜色多根线
line_width = [3, 3, 2, 2, 3, 2, 2, 2]
color = ['limegreen','sienna', 'royalblue','purple', 'orangered', 'hotpink', 'black', 'grey',   'm', 'm', 'y', 'k', 'grey', ]
line_style = ['--', '--', '--', '--', '--', '--', '--', '--', '--', '.-', '.-']
dash_style = [[1,0], [1,1], [1,0], [1,1], [1,0], [1,1], [1,0], [1,0],]

draw_idx = 0  ## 曲线个数，画几条线
label = ['DQN', '0.1', '0.2']   ## 标签

# line_width = [3]
# color = ['limegreen']
# line_style = ['--']
# dash_style = [[1,0]]
#
# draw_idx = 0  ## 曲线个数，画几条线
# label = ['DQN']   ## 标签

## 画曲线
# plt.figure()


# plt.plot(x, result_qmix_mean, line_style[draw_idx], dashes=dash_style[draw_idx], color=color[draw_idx], label=label[draw_idx], linewidth=line_width[draw_idx])

## 填充阴影
# plt.fill_between(x, result_qmix_min, result_qmix_max, facecolor=color[draw_idx], alpha=0.1)


## 平滑后输出
plt.plot(xs*20, y_noise01, line_style[draw_idx], dashes=dash_style[draw_idx], color=color[draw_idx], label=label[draw_idx], linewidth=line_width[draw_idx])
#plt.fill_between(xs*20, y_min_qmix, y_max_qmix, facecolor=color[draw_idx], alpha=0.1)
draw_idx += 1

# plt.plot(x, result_iql_mean, line_style[draw_idx], dashes=dash_style[draw_idx], color=color[draw_idx], label=label[draw_idx], linewidth=line_width[draw_idx])
# plt.fill_between(x, result_iql_min, result_iql_max, facecolor=color[draw_idx], alpha=0.1)
plt.plot(xs*20, y_noise11, line_style[draw_idx], dashes=dash_style[draw_idx], color=color[draw_idx], label=label[draw_idx], linewidth=line_width[draw_idx])
#plt.fill_between(xs*20, y_min_iql, y_max_iql, facecolor=color[draw_idx], alpha=0.1)
draw_idx += 1

# plt.plot(x, result_vdn_mean, line_style[draw_idx], dashes=dash_style[draw_idx], color=color[draw_idx], label=label[draw_idx], linewidth=line_width[draw_idx])
# plt.fill_between(x, result_vdn_min, result_vdn_max, facecolor=color[draw_idx], alpha=0.1)
plt.plot(xs*20, y_noise21, line_style[draw_idx], dashes=dash_style[draw_idx], color=color[draw_idx], label=label[draw_idx], linewidth=line_width[draw_idx])
#plt.fill_between(xs*20, y_min_vdn, y_max_vdn, facecolor=color[draw_idx], alpha=0.1)
draw_idx += 1


#plt.plot(xs, nosie22, line_style[draw_idx], dashes=dash_style[draw_idx], color=color[draw_idx], label=label[draw_idx], linewidth=line_width[draw_idx])
#draw_idx += 1

## 绘制网格
plt.grid(linewidth=0.5)
plt.grid(linewidth=0.5)

## 横坐标显示间隔
# plt.xticks(x[::INTERNAL*2])
# plt.xticks(x[::INTERNAL*10])

# ax_ = plt
#!*******************************************边框粗细
def set_border(ax_):
    bwith = 0.5
    ax_.spines['bottom'].set_linewidth(bwith)
    ax_.spines['left'].set_linewidth(bwith)
    ax_.spines['top'].set_linewidth(bwith)
    ax_.spines['right'].set_linewidth(bwith)

set_border(ax)

#!*******************************************边框粗细end



## 标签位置设置
# plt.legend(
#     # loc='upper left',
#     # loc=0,   ## 自动设置标签位置
#     markerscale=5.0, ncol=3, columnspacing=3,
#     framealpha=0.9, prop=legend_front,
#     bbox_to_anchor=(0.95, 1.21)    ## 控制标签位置，第一个数控制左右移动，第二个数控制上下移动
#
#           )

plt.legend(loc='lower right', markerscale=1.0, ncol=3, columnspacing=2, framealpha=0.9, prop=legend_front)


## 横纵坐标名称

plt.tick_params(labelsize=14)

# plt.xlabel('训练回合数', fontdict=xy_descri)
# plt.ylabel('平均奖励', fontdict=xy_descri)

plt.xlabel('Number of training episodes', fontdict=xy_descri)
plt.ylabel('Total long-term reward', fontdict=xy_descri)

# plt.show()
# plt.savefig("3uav_0.8.png", bbox_inches='tight')
# plt.savefig("3uav_0.8.svg", bbox_inches='tight')
# plt.savefig("3uav_0.8.pdf", bbox_inches='tight')
# plt.savefig("3uav_0.8.eps", bbox_inches='tight')

plt.savefig("3uav_0.8_.png", bbox_inches='tight')
plt.savefig("3uav_0.8_.svg", bbox_inches='tight')
plt.savefig("3uav_0.8_.pdf", bbox_inches='tight')
plt.savefig("3uav_0.8_.eps", bbox_inches='tight')


## 保存文件
save_path = './'
# plt.savefig(save_path + 'qmix.png', format='png')
# plt.close()

# np.save(save_path + '/qmix_mean.npy', Mean)
# np.savetxt(save_path + '/qmix_mean.txt', Mean, fmt='%.4f', delimiter='\n')
