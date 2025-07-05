from tensorflow.keras.datasets import mnist
import numpy as np

def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    X_train, X_test = X_train.reshape(-1, 28 * 28), X_test.reshape(-1, 28 * 28)
    y_train_enc = one_hot_encode(y_train)
    y_test_enc = one_hot_encode(y_test)
    return X_train, y_train_enc, X_test, y_test_enc, y_test

def one_hot_encode(y):
    encoded = np.zeros((y.shape[0], 10))
    encoded[np.arange(y.shape[0]), y] = 1
    return encoded
