import numpy as np
import os, sys
sys.path.append(os.pardir)
from mnist import load_mnist
import pickle

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    if os.path.exists("weight.pkl"):
        with open("weight.pkl", 'rb') as f:
            network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def softmax(X):
    m = np.max(X)
    exp_X = np.exp(X - m)
    sum_exp_X = np.sum(exp_X)
    return exp_X / sum_exp_X

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) + f(x-h)) / (2 * h)

x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in rainge(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print ("Accuracy:" + str(float(accuracy_cnt) / len(x)))
