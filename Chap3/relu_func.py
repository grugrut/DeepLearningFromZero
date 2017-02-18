import numpy as np
import matplotlib.pyplot as plt

def ReLU(x):
    return np.maximum(x, 0)

x = np.arange(-5, 5, 0.1)
y = ReLU(x)

plt.plot(x, y)
plt.ylim(-0.1, 6)
plt.show()
