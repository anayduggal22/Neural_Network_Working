from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Keras Neural Network

model = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(2,activation='sigmoid'),
    layers.Dense(1,activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.5),
    loss = 'mse'
)

# Creating a tiny dataset [XOR Operation]

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model.fit(X, y, epochs=6000, verbose=0)

predictions = model.predict(X, verbose=0)

print("\nFinal Predictions:")
print(predictions.round(3))

print("\nTrue Labels:")
print(y)