import numpy as np

def sigmoid(z):
   
    return 1 / (1 + np.exp(-z))
def initialize_weights(n):

    return np.zeros((n, 1))
def compute_loss(y, y_hat):
    m = y.shape[0]
    epsilon = 1e-15  
   
   
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    return -(1/m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def logistic_regression(X, y, learning_rate=0.1, epochs=1000):
    m, n = X.shape
 
    X = np.hstack((np.ones((m, 1)), X))  
    w = initialize_weights(n + 1)

    for epoch in range(epochs):
        z = np.dot(X, w)
        y_hat = sigmoid(z)
        loss = compute_loss(y, y_hat)

        
        gradient = (1/m) * np.dot(X.T, (y_hat - y))

       
        w -= learning_rate * gradient

        if epoch % 100 == 0:
            print(f"Epoch {epoch} => Loss: {loss:.4f}")
    
    return w


def predict(X, w):
    
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))  
   
    y_prob = sigmoid(np.dot(X, w))
    return (y_prob >= 0.5).astype(int)

# example
X_train = np.array([[0.1, 0.6],
                    [0.2, 0.7],
                    [0.4, 0.5],
                    [0.9, 0.4],
                    [1.0, 0.3]])
y_train = np.array([[0], [0], [0], [1], [1]])


weights = logistic_regression(X_train, y_train, learning_rate=0.1, epochs=1000)


preds = predict(X_train, weights)

print("\nPredictions:", preds.ravel())  # Flatten for display






