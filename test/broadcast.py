import numpy as np

grads = np.random.randint(1, 50, (4, 3))
weight = np.random.random(3)

print(grads)
print(weight)
print(grads.dot(weight))