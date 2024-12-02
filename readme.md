# Randria Linear Regression

This Python code implements a custom linear regression model using gradient descent. It generates a synthetic dataset, builds a linear regression model, and trains it using gradient descent.

## Features

- **Dataset Creation**: 
  - The dataset consists of 100 samples with 2 features.
  - Targets are generated based on a linear regression model with added noise.
  
- **Linear Regression Model**: 
  - Implements a custom linear regression model using the formula \( y = X \cdot \theta \), where \( X \) is the input features and \( \theta \) is the model parameters.
  - The model uses gradient descent for optimization.
  
- **Gradient Descent**: 
  - A function to compute the gradient of the cost function with respect to the parameters \( \theta \).
  - The model is trained by iteratively updating the parameters using the gradient and a learning rate.

- **Loss Calculation**: 
  - Computes the Mean Squared Error (MSE) as the loss function.
  
- **Training and Visualization**: 
  - After training, the loss curve is plotted to show the progression of the model's performance over iterations.

## Dependencies

- `numpy`
- `matplotlib`
- `scikit-learn`

## Code Breakdown

### Data Generation

```python
from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)
x, y = make_regression(n_samples=100, n_features=2, noise=10)
plt.scatter(x[:, 1], y)
plt.show()

print(x.shape)  # (100, 2)
print(y.shape)  # (100,)
```

- A synthetic regression dataset is generated with 100 samples and 2 features.
- The target variable is computed with added noise.

### Linear Regression Model Class

```python
class RandriaLineaReg:
    lossAt = []

    def reshapeFeat(x):
        X = np.hstack((x, np.ones((x.shape[0], 1))))  # Add bias term
        return X

    theta = np.random.randn(3, 1)  # Initialize parameters

    def model(X):
        return X.dot(RandriaLineaReg.theta)

    def grad(X, y):
        return (1 / len(y)) * X.T.dot(RandriaLineaReg.model(X) - y)

    def fit(X, y, learningRate, iteration):
        for i in range(iteration):
            RandriaLineaReg.theta = RandriaLineaReg.theta - learningRate * (RandriaLineaReg.grad(X, y))
            RandriaLineaReg.lossAt.append(RandriaLineaReg.costFunction(RandriaLineaReg.predict(X), y))

    def predict(X):
        return RandriaLineaReg.model(X)

    def costFunction(model, y):
        return (1 / y.shape[0]) * np.sum((model - y) ** 2)
```

- The `RandriaLineaReg` class encapsulates the linear regression model.
- The model includes methods for reshaping features, predicting outputs, calculating gradients, and updating parameters using gradient descent.
  
### Model Training

```python
X = RandriaLineaReg.reshapeFeat(x)
model = RandriaLineaReg.model(X)
print("Initial LOSS : {}".format(RandriaLineaReg.costFunction(model, y)))

RandriaLineaReg.fit(X, y, 0.01, 1000)
print("Trained theta: {}".format(RandriaLineaReg.theta))
ypred = RandriaLineaReg.predict(X)

print("LOSS AFTER TRAINING : {}".format(RandriaLineaReg.costFunction(ypred, y)))
```

- The model is initialized and trained with a learning rate of 0.01 for 1000 iterations.
- Loss is calculated before and after training to observe the performance improvement.

### Visualization

```python
plt.plot(RandriaLineaReg.lossAt)
plt.show()
```

- The loss curve is plotted during training to visualize the model's convergence.

## Example Output

```text
Initial LOSS: 10456.314962870447
Trained theta: [[ 84.79379536]
                 [ 52.15488894]
                 [-0.28606427]]
LOSS AFTER TRAINING: 80.7041521951798
```

## Conclusion

This code demonstrates a custom implementation of linear regression using gradient descent. The model is trained on a synthetic dataset, and the performance is evaluated by plotting the loss over iterations.
