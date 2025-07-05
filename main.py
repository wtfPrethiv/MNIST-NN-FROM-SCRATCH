from models.neural_network import NeuralNetwork
from utils.dataset_loader import load_mnist
from utils.metrics import accuracy
import matplotlib.pyplot as plt
from utils.visualization import show_single_image

X_train, y_train_enc, X_test, y_test_enc, y_test = load_mnist()

model = NeuralNetwork()
model.train(X_train, y_train_enc, epochs=1000, lr=0.03)

acc = model.accuracy(X_test, y_test_enc)
print(f"Test Accuracy: {acc*100:.2f}%")

# Visualization
index = 43
image = X_test[index]
true_label = y_test[index]
pred = model.predict(image.reshape(1, -1))
predicted_label = pred.argmax()

show_single_image(image, label=true_label, prediction=predicted_label)

pred = model.predict(X_test[index].reshape(1, -1))
print("Predicted:", pred[0])
