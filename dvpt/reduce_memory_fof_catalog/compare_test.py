import numpy as np

test1 = np.load('test1.npy')
test3 = np.load('test3.npy')

s = test1 == test3

print(f"There is {s.sum()} True on {s.size} --> {s.sum()/s.size:2.2%}")
