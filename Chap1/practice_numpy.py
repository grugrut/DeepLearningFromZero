import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])

# 要素ごとの四則演算

print ("x + y = %s" % (x + y))
print ("x - y = %s" % (x - y))
print ("x * y = %s" % (x * y))
print ("x / y = %s" % (x / y))

# N次元配列の場合

x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

print ("x + y = %s" % (x + y))
print ("x - y = %s" % (x - y))
print ("x * y = %s" % (x * y))
print ("x / y = %s" % (x / y))

#要素へのアクセス
X = np.array([[1, 2], [10, 20], [100, 200]])

for row in X:
    print(row)

print (X > 15)
print (X[X > 15])

