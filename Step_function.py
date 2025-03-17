"""

# np.array, arange 학습 필요.

*기초 계단함수 구현*

import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    if x > 0:
        return 1
    else:
        return 0


def step_function(x):
    y = x > 0
    return y.astype(np.int)

x = np.array([-1.0, 1.0, 2.0])
y = x > 0

y.astype(np.int)

import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    
    return np.array(x > 0, dtype=np.int32)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)    

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show() 

"""
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    
    return np.array(x > 0, dtype=np.int32)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)    

plt.plot(x, y)
plt.ylim(-0.1, 1.1) #y축 범위 지정
plt.show()

y = sigmoid(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

y = relu(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

