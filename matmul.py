import numpy as np 
import time 

x = np.random.rand(1024, 1024)
y = np.random.rand(1024, 1024)

st = time.monotonic()
z = np.dot(x, y)
print("np.dot: ", time.monotonic() - st)

st = time.monotonic()
z = x @ y
print("@: ", time.monotonic() - st)