import numpy as np

A = np.array([1, 2, 3, 4])
B = np.array([[1, 2], [3, 4], [5, 6]])

print(A)
print(np.ndim(A))
print(A.shape)

print(B)
print(np.ndim(B))
print(B.shape)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.dot(A, B))

A = np.array([1, 2, 3])
B = np.array([4, 5, 6])
print(np.dot(A, B))
