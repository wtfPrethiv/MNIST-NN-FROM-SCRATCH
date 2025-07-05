import numpy as np
np.random.seed(55)

class NeuralNetwork:

    def __init__(self):
        self.w = None
        self.b = None

    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) 
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def dense(self, a_in, w, b, activation='relu'):
        z = np.dot(a_in, w) + b
        if activation == 'relu':
            a_out = self.relu(z)
        elif activation == 'softmax':
            a_out = self.softmax(z)
        return z, a_out

    def sequential(self, a_in, w, b):
        z1, a_1 = self.dense(a_in, w[0], b[0], activation='relu') 
        z2, a_2 = self.dense(a_1, w[1], b[1], activation='relu')
        z3, a_3 = self.dense(a_2, w[2], b[2], activation='softmax')
        return z1, a_1, z2, a_2, z3, a_3
    
    def categorical_cross_entropy_loss(self, y, y_hat):  
        eps = 1e-9
        m = y.shape[0]
        loss = -np.sum(y * np.log(y_hat + eps)) / m   
        return loss
    
    def one_hot(self, y, num_classes=10):
        m = y.shape[0]
        one_hot_y = np.zeros((m, num_classes))
        one_hot_y[np.arange(m), y] = 1
        return one_hot_y
    
    def train(self, X, y, epochs, lr):
        samples, features = X.shape

         
        self.w = [
            np.random.randn(features, 128) * np.sqrt(2. / features),
            np.random.randn(128, 64) * np.sqrt(2. / 128),
            np.random.randn(64, 10) * np.sqrt(2. / 64)
        ]

        self.b = [
            np.zeros((1, 128)),
            np.zeros((1, 64)),
            np.zeros((1, 10))
        ]

        for i in range(epochs):
            z1, a_1, z2, a_2, z3, a_3 = self.sequential(X, self.w, self.b)

            loss = self.categorical_cross_entropy_loss(y, a_3)   
            
            m = y.shape[0]

        
            dz3 = (a_3 - y) / m   

            dw3 = np.dot(a_2.T, dz3)
            db3 = np.sum(dz3, axis=0, keepdims=True)

            da2 = np.dot(dz3, self.w[2].T)
            dz2 = da2 * self.relu_derivative(z2)

            dw2 = np.dot(a_1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)

            da1 = np.dot(dz2, self.w[1].T)
            dz1 = da1 * self.relu_derivative(z1)

            dw1 = np.dot(X.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)

             
            self.w[2] -= lr * dw3
            self.b[2] -= lr * db3

            self.w[1] -= lr * dw2
            self.b[1] -= lr * db2

            self.w[0] -= lr * dw1
            self.b[0] -= lr * db1

            if i % 100 == 0:
                print(f"Epoch {i}: Loss = {loss:.4f}")
    
    def accuracy(self, X, y_true):
        y_pred = self.predict(X)
        y_true_labels = np.argmax(y_true, axis=1)
        acc = np.mean(y_pred == y_true_labels)
        return acc


    def predict(self, X):
        _, _, _, _, _, a3 = self.sequential(X, self.w, self.b)
        return np.argmax(a3, axis=1)