import os

from mindl.nn import NeuralNetwork
from mindl.function.loss import MSE
from mindl.function.activation import ReLU
import numpy as np

mnist_dir = 'data/mnist'
if not os.path.exists(mnist_dir):
    import urllib.request
    import gzip

    os.mkdir(mnist_dir)
    files_to_download = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    ]
    for url in files_to_download:
        with urllib.request.urlopen(url) as gz_file:
            zip_file = os.path.join(mnist_dir, os.path.basename(url))
            with open(zip_file, 'wb') as f:
                f.write(gz_file.read())

        with gzip.open(zip_file, 'r') as f_in:
            if 'labels' in zip_file:
                # second 4 bytes is the number of labels
                label_count = int.from_bytes(f_in.read(4), 'big')
                # rest is the label data, each label is stored as unsigned byte
                # label values are 0 to 9
                label_data = f_in.read()
                data = np.frombuffer(label_data, dtype=np.uint8)
            elif 'images' in zip_file:
                # first 4 bytes is a magic number
                magic_number = int.from_bytes(f_in.read(4), 'big')
                # second 4 bytes is the number of images
                image_count = int.from_bytes(f_in.read(4), 'big')
                # third 4 bytes is the row count
                row_count = int.from_bytes(f_in.read(4), 'big')
                # fourth 4 bytes is the column count
                column_count = int.from_bytes(f_in.read(4), 'big')
                # rest is the image pixel data, each pixel is stored as an unsigned byte
                # pixel values are 0 to 255
                image_data = f_in.read()
                data = np.frombuffer(image_data, dtype=np.uint8).reshape((image_count, row_count, column_count))

            np.save(zip_file.replace('.gz', '.npy'), data)
        os.remove(zip_file)

test_X = np.load('data/mnist/t10k-images-idx3-ubyte.npy')
train_X = np.load('data/mnist/train-images-idx3-ubyte.npy')

test_X = np.where(test_X == 0, test_X, 1)
test_X = test_X.reshape(test_X.shape[0], -1)
train_X = np.where(train_X == 0, train_X, 1)
train_X = train_X.reshape(train_X.shape[0], -1)

test_y = np.load('data/mnist/t10k-labels-idx1-ubyte.npy')
train_y = np.load('data/mnist/train-labels-idx1-ubyte.npy')

y_max = test_y.max() + 1
test_y = np.eye(y_max)[test_y]
train_y = np.eye(y_max)[train_y]


# todo: add minibatch
# todo: add weight decay
nn = NeuralNetwork(shape=[test_X.shape[1], 16, 16, test_y.shape[1]], learning_rate=0.01, activation=ReLU(), loss=MSE(), log_frequency=10)

nn.fit(train_X, train_y, 1000)

print(
    f"Ground-truth: {train_y[:10]}"
)
print(
    f"Predicted: {np.rint(nn.forward(train_X[:10]))}"
)

