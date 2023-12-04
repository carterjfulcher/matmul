"""
Strassen algorithm is apparently much better. From N^3 flops to N2.81 
"""

import numpy as np

N = 8

x = np.random.rand(N, N)
y = np.random.rand(N, N)

def strassen_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
  A = [A[i:i+N//2, j:j+N//2] for i in range(0, N, N//2) for j in range(0, N, N//2)]
  B = [B[i:i+N//2, j:j+N//2] for i in range(0, N, N//2) for j in range(0, N, N//2)]

  # calculate sub matrices 
  p_1 = A[1][1] * (B[1][2] - B[2][2])
  p_2 = (A[1][1] * A[1][2]) * B[2][2]
  p_3 = (A[2][1] + A[2][2]) * B[1][1]
  p_4 = A[2][2] * (B[2][1] - B[1][1])
  p_5 = (A[1][1] + A[2][2]) * (B[1][1] + B[2][2])
  p_6 = (A[1][2] - A[2][2]) * (B[2][1] + B[2][2])
  p_7 = (A[1][1] - A[2][1]) * (B[1][1] + B[1][2])

  # create result submatrices
  C = [np.zeros((N, N))[i:i+N//2, j:j+N//2] for i in range(0, N, N//2) for j in range(0, N, N//2)]
  C[1][1] = p_5 + p_4 - p_2 + p_6
  C[1][2] = p_1 + p_2
  C[2][1] = p_3 + p_4
  C[2][2] = p_1 + p_5 - p_3 - p_7

  # combine result submatrices
  rows = [np.hstack(C[i:i+2]) for i in range(0, len(C), 2)]
  return np.vstack(rows)

C = strassen_matmul(x, y)

C_numpy = x@y

print(C)

print(C_numpy)

print(C.all() == C_numpy.all())

