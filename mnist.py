import urllib.request
import os.path
import gzip
import pickle
import numpy as np

mnist_url_base = 'http://yann.lecun.com/exdb/mnist/'
mnist_file = {
    'train_img'   : 'train-images-idx3-ubyte.gz',
    'train_label' : 'train-labels-idx1-ubyte.gz',
    'test_img'       : 't10k-images-idx3-ubyte.gz',
    'test_label'     : 't10k-labels-idx1-ubyte.gz'}

data_dir = os.path.dirname(os.path.abspath(__file__))
save_file = data_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

def _download(filename):
    filepath = data_dir + "/" + filename

    if os.path.exists(filename):
        return

    print("Download " + filename)
    urllib.request.urlretrieve(mnist_url_base + filename, filepath)
    print("Done")

def download_mnist():
    for filename in mnist_file.values():
        _download(filename)

def _load_label(filename):
    filepath = data_dir + "/" + filename

    with gzip.open(filepath, 'rb') as f:
        label = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return label

def _load_img(filename):
    filepath = data_dir + "/" + filename

    with gzip.open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)        
    data = data.reshape(-1, img_size)
    print("Done")

    return data
    
def _convert_numpy():
    dataset = {}

    for key in ('train_img', 'test_img'):
        dataset[key] = _load_img(mnist_file[key])

    for key in ('train_label', 'test_label'):
        dataset[key] = _load_label(mnist_file[key])

    return dataset

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)

def _change_one_hot_label(x):
    T = np.zeros((x.size, 10))
    for idx, row in enumerate(T):
        row[x[idx]] = 1

    return T

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """
    MNISTデータセットを読み込む
    normalize 正規化
    flatten 一次元配列にする
    one_host_label ラベルをone-hot配列([1,0,0,0,0,0,0,0,0])にする
    """

    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        for key in ('train_label', 'test_label'):
            dataset[key] = _change_one_hot_label(dataset[key])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

if __name__ == '__main__':
    init_mnist()
