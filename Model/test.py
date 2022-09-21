import tensorflow as tf

a = [1, 2, 3]
b = [4, 5, 6]

import numpy as np

print(np.random.randint(0, 100, (10, 100)).shape)

inputs = tf.random.normal([32, 10, 8])
lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True)
output, state_h, state_c = lstm(inputs)

print("output", output)
print("state_h", state_h)

if output[]