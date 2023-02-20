import numpy as np

x = np.array([1, 2, 3])
print(x.__class__)
print(x.shape)
print(x.ndim)

W = np.array([[1, 2, 3], [4, 5, 6]])
print(W.__class__)
print(W.shape)
print(W.ndim)

X = np.array([[0, 1, 2], [3, 4, 5]])
print(W + X)
print(W * X)

A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
print(A * B)

y = np.array([4, 5, 6])
print(np.dot(x, y))
print(np.matmul(A, W))
