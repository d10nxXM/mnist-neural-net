import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv("mnist_train.csv")

train_data = np.array(train_data)
np.random.shuffle(train_data)

m_train, n = train_data.shape
train_data_T = train_data.T
Y_train = train_data_T[0]
X_train = train_data_T[1:n] / 255

def init_params():
    w1 = np.random.randn(10, 784) * 0.01
    b1 = np.zeros((10, 1))
    w2 = np.random.randn(10, 10) * 0.01
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    exp = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp / exp.sum(axis=0, keepdims=True)

def forward_propagation(w1, b1, w2, b2, X):
    Z1 = w1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = w2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.max() + 1, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def der_ReLU(Z):
    return Z > 0

def backward_propagation(Z1, A1, Z2, A2, w1, w2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = w2.T.dot(dZ2) * der_ReLU(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_parameters(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha):
    w1 -= alpha * dW1
    b1 -= alpha * db1
    w2 -= alpha * dW2
    b2 -= alpha * db2
    return w1, b1, w2, b2

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def save_model(w1, b1, w2, b2):
    np.save("weights_w1.npy", w1)
    np.save("weights_b1.npy", b1)
    np.save("weights_w2.npy", w2)
    np.save("weights_b2.npy", b2)
    print("Model saved successfully!")

def load_model():
    w1 = np.load("weights_w1.npy")
    b1 = np.load("weights_b1.npy")
    w2 = np.load("weights_w2.npy")
    b2 = np.load("weights_b2.npy")
    print("Model loaded successfully!")
    return w1, b1, w2, b2

def gradient_descent(X, Y, alpha, iterations):
    w1, b1, w2, b2 = init_params()
    # w1, b1, w2, b2 = load_model()  
    accuracy_list = []

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(w1, b1, w2, b2, X)
        dW1, db1, dW2, db2 = backward_propagation(Z1, A1, Z2, A2, w1, w2, X, Y)
        w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha)

        if i % 100 == 0:  
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            accuracy_list.append(accuracy)
            print(f"Iteration: {i}, Accuracy: {accuracy:.4f}")

    save_model(w1, b1, w2, b2)

    plt.plot(range(0, iterations, 50), accuracy_list)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Over Time")
    plt.show()

    return w1, b1, w2, b2

w1, b1, w2, b2 = gradient_descent(X_train, Y_train, 0.05, 1500)

# To load the trained model later:
# w1, b1, w2, b2 = load_model()



w1, b1, w2, b2 = load_model()

Z1_test, A1_test, Z2_test, A2_test = forward_propagation(w1, b1, w2, b2, X_train)

predictions_test = get_predictions(A2_test)

print("Misclassified Samples:")
for i in range(20):  
    actual_label = Y_train[i]
    predicted_label = predictions_test[i]
    
    if actual_label != predicted_label: 
        print(f"Test Sample {i+1}: Expected: {actual_label}, Predicted: {predicted_label}")

        plt.imshow(X_train[:, i].reshape(28, 28), cmap='gray')
        plt.title(f"Incorrect Prediction\nActual: {actual_label}, Predicted: {predicted_label}")
        plt.axis("off")
        plt.show()
    else:
        print(f"Test Sample {i+1}: Expected: {actual_label}, Predicted: {predicted_label}")
        
        
accuracy_test = get_accuracy(predictions_test, Y_train)