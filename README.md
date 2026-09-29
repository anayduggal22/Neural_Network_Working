# Neural Network From Scratch (NumPy)

A minimal neural network implemented entirely from scratch in NumPy 
forward pass, backpropagation, and gradient descent with no deep learning framework verified against an equivalent
Keras model trained on the same data.

------------------------------------------------------------------------

## Table of Contents

-   Overview
-   Motivation
-   The Problem: XOR
-   Network Architecture
-   Implementation Details
-   Verification Against Keras
-   Results
-   Key Findings
-   Tech Stack
-   Project Structure
-   How to Run
-   Author

------------------------------------------------------------------------

## Overview

This project implements a small neural network's forward pass,
backpropagation, and gradient descent update rule entirely by hand in
NumPy, without using any deep learning framework. The goal was to move
past using `model.fit()` as a black box and actually understand the
mechanics of how a neural network learns.

The same architecture was then reproduced in Keras and trained on
identical data, to confirm the from-scratch implementation is
mathematically equivalent to what a standard framework computes
internally.

------------------------------------------------------------------------

## The Problem: XOR

The XOR logic problem was used as the test case:

| Input | Output |
|---|---|
| (0, 0) | 0 |
| (0, 1) | 1 |
| (1, 0) | 1 |
| (1, 1) | 0 |

XOR is a deliberate choice: it is not linearly separable, so a network
with no hidden layer cannot solve it. Successfully solving XOR is
therefore direct evidence that the hidden layer and backpropagation
implementation are both working correctly, not just running without
errors.

------------------------------------------------------------------------

## Network Architecture

```text
Input (2) -> Dense(2, sigmoid) -> Dense(1, sigmoid)
```

-   2 input features
-   1 hidden layer with 2 neurons, sigmoid activation
-   1 output neuron, sigmoid activation
-   Loss function: Mean Squared Error
-   Optimizer: plain gradient descent, learning rate 0.5
-   Trained for 6,000 epochs

------------------------------------------------------------------------

## Implementation Details

**Forward pass:**

```python
z1 = X @ w1 + b1
h = sigmoid(z1)

z2 = h @ w2 + b2
o = sigmoid(z2)
```

**Backward pass (backpropagation via the chain rule):**

```python
d_loss_o = -2 * (y - o)
d_o_z2 = sigmoid_derivative(o)
d_z2 = d_loss_o * d_o_z2

dw2 = h.T @ d_z2
db2 = np.sum(d_z2, axis=0, keepdims=True)

d_h = d_z2 @ w2.T
d_h_z1 = sigmoid_derivative(h)
d_z1 = d_h * d_h_z1

dw1 = X.T @ d_z1
db1 = np.sum(d_z1, axis=0, keepdims=True)
```

**Gradient descent update:**

```python
w2 = w2 - lr * dw2
b2 = b2 - lr * db2
w1 = w1 - lr * dw1
b1 = b1 - lr * db1
```

The gradient for the output layer is computed first, then propagated
backward into the hidden layer, reusing the already computed output
layer gradient rather than recalculating it from the loss directly.
This is the core mechanic of backpropagation: computing gradients in
reverse order (output to input) so each layer's local derivative is
reused rather than recomputed.

------------------------------------------------------------------------

## Verification Against Keras

An equivalent model was built in Keras, using the same architecture,
activation functions, loss function, optimizer, learning rate, and
epoch count, trained on the identical XOR dataset:

```python
model = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(2, activation='sigmoid'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.5),
    loss='mse'
)

model.fit(X, y, epochs=6000, verbose=0)
```

------------------------------------------------------------------------

## Results

| | Predicted (0,0) | Predicted (0,1) | Predicted (1,0) | Predicted (1,1) |
|---|---|---|---|---|
| True label | 0 | 1 | 1 | 0 |
| From-scratch NumPy | 0.023 | 0.980 | 0.980 | 0.024 |
| Keras (equivalent architecture) | 0.043 | 0.964 | 0.962 | 0.038 |

Both implementations converged to nearly identical predictions,
correctly separating the XOR classes. The small numerical differences
between the two are expected, and come from differing random weight
initialization and internal implementation details of Keras's
optimizer, not from any error in the from-scratch version.

------------------------------------------------------------------------

## Key Findings

-   A from-scratch NumPy implementation of backpropagation, using only
    the chain rule and gradient descent, converges to the same result
    as an equivalent Keras model trained under the same conditions.
    This confirms the manual implementation is mathematically correct.
-   At very small epoch counts (around 1,000), the two implementations
    can appear meaningfully different in their predictions, since
    neither has fully converged yet. This is a training-time artifact,
    not a discrepancy in the underlying math both converge to
    equivalent results once trained sufficiently.
-   For a dataset this small (4 samples), Keras's per-epoch overhead
    (graph tracing, logging, callback checks) is disproportionately
    large relative to the trivial amount of actual computation
    involved. A short training run of very few epochs can appear to
    take a noticeable amount of time in Keras, purely from this fixed
    overhead, while the equivalent NumPy loop runs near instantly.
    Deep learning frameworks are optimized for scale, not
    toy sized problems.

------------------------------------------------------------------------

## Tech Stack

-   Python
-   NumPy
-   TensorFlow / Keras (for verification only)

------------------------------------------------------------------------

## Project Structure

```text
neural-network-from-scratch/
├── Working.py
├── Keras_NN.py
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

## How to Run

```bash
git clone https://github.com/anayduggal22/Neural_Network_Working

cd Neural_Network_Working

python -m venv venv
venv\Scripts\activate
pip install numpy

python Working.py
```

The Keras verification script requires a separate environment with
TensorFlow installed:

```bash
py -3.12 -m venv venv312
venv312\Scripts\activate
pip install -r requirements.txt

python Keras_NN.py
```

------------------------------------------------------------------------

## Author

**Anay Duggal**
