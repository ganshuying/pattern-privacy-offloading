import math

B = 5  # 5000KHz/5MHz
ptr = 0.2  # 23dBm/0.2w
gs = 1.0e-8
N0 = 3.9e-18
part = 1 + ((ptr*gs*gs)/(N0*B))

rtr = B * math.log2(part) # 5MB/s

print(rtr)