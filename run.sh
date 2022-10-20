nohup python3 new_run.py --sigma=0.5 --num_episodes=5000 >&1 &




# noisy=0.1

# 已坐
nohup python3 new_run.py --sigma=0.1 --num_episodes=1500 --edge_computing_capacity=100 > 0.1_100.log 2>&1 &
nohup python3 new_run.py --sigma=0.1 --num_episodes=1500 --edge_computing_capacity=200 > 0.1_200.log 2>&1 &

#