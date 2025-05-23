import numpy as np


def step_function(z):
    return 1 if z > 0 else 0


def perceptron_predict(X, w, b):
    return step_function(np.dot(X, w) + b)


def train_perceptron(X, y, lr=0.1, epochs=100):
    
    
    w = np.zeros(X.shape[1])  
    b = 0                     
    for _ in range(epochs):
        
        for xi, target in zip(X, y):
            y_pred = perceptron_predict(xi, w, b)
            error = target - y_pred
            w += lr * error * xi
            b += lr * error
    return w, b


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])  


w, b = train_perceptron(X, y)

print("Trained weights:", w)
print("Trained bias:", b)


for xi in X:
    print(f"Input: {xi}, Prediction: {perceptron_predict(xi, w, b)}")
