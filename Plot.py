from cProfile import label
import matplotlib.pyplot as plt
import numpy as np

loss = np.load('losses.npy')
acc = np.load('accuracies.npy')

plt.plot(loss, label='loss')
plt.plot(acc, label='accuracy')
plt.legend()
plt.show()
