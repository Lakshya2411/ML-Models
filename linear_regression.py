X = [1, 2, 3, 4, 5]
Y = [2, 4, 5, 4, 5]

def mean(values):
    mean=sum(values)/len(values)
    return mean 

def coefficients(X,Y):
    x_mean=mean(X)
    y_mean=mean(Y)
    numerator = 0
    denominator = 0


    for i in range(len(X)):
        numerator += (X[i] - x_mean) * (Y[i] - y_mean)
        denominator += (X[i] - x_mean) ** 2
    beta_1=numerator/denominator
    beta_0=y_mean-beta_1*x_mean
    return beta_0,beta_1


def predict(X, beta_0, beta_1):
    predictions = []
    for x in X:
        y_pred = beta_0 + beta_1 * x
        predictions.append(y_pred)
    return predictions


X = [1, 2, 3, 4, 5]
Y = [2, 4, 5, 4, 5]


b0, b1 = coefficients(X, Y)
print(f"Model: Y = {b0:.2f} + {b1:.2f} * X")


Y_pred = predict(X, b0, b1)
print("Predictions:", Y_pred)


    