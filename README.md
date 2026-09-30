# ML From Scratch

Classic machine learning algorithms implemented **from their mathematical definitions**, using only Python and NumPy — no scikit-learn, TensorFlow or PyTorch. The goal is to understand what libraries do under the hood.

## Implemented

| Algorithm | File | How it's implemented |
|---|---|---|
| Linear Regression | `linear_regression.py` | Closed-form least squares: slope = covariance(X, Y) / variance(X), intercept from the means. Pure Python, no libraries |
| Logistic Regression | `logistic_regression.py` | Sigmoid + binary cross-entropy loss, trained with batch gradient descent. Bias handled by adding a column of ones |
| Perceptron | `perceptron.py` | Step activation with the perceptron learning rule (`w += lr × error × x`), trained on the AND gate |

## Run them

```bash
pip install numpy
python linear_regression.py
python logistic_regression.py
python perceptron.py
```

Example output:

```text
$ python linear_regression.py
Model: Y = 2.20 + 0.60 * X

$ python logistic_regression.py
Epoch 900 => Loss: 0.1196
Predictions: [0 0 0 1 1]

$ python perceptron.py
Input: [1 1], Prediction: 1      # learned the AND gate
```

## Planned next

- K-Nearest Neighbors
- K-Means clustering
- Naive Bayes
- Principal Component Analysis (PCA)
- Decision Tree
- Support Vector Machine
- Neural network with one hidden layer (backpropagation)
