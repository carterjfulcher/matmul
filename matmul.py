import numpy as np 
import time 

N = 8

x = np.random.rand(N, N)
y = np.random.rand(N, N)

st = time.monotonic()
z = x @ y

at_time = time.monotonic() - st

print("@: ", at_time)

""" Naive matmul """
def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
  # A is size m x n, B is size n x p. C (output) is size m x p. Multiply->Add
  assert A.shape[1] == B.shape[0]
  C, n = np.zeros((A.shape[0], B.shape[1])), A.shape[1] 
  for i in range(A.shape[0]):
    for j in range(B.shape[0]):
      C[i][j] = sum([A[i][k] * B[k][j] for k in range(n)])

  return C

st = time.monotonic()
C = matmul(x, y)
matmul_time = time.monotonic() - st
print(f"matmul: {matmul_time}")

exp = C.all() == z.all()
print(f"CHECK: {exp}")
assert exp

print(f"@ is {matmul_time/at_time:.2f} faster than matmul")