import sys
import os

# dataset 폴더가 포함된 'deep-learning-from-scratch-master' 경로를 추가
sys.path.append("C:/Users/Lenovo/Desktop/code/Git/AI-from-basic/deep-learning-from-scratch-master")

# 이제 dataset.mnist를 임포트할 수 있음
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten = True, normalize=False)

print(x_train.shape) #(60000, 784)
print(t_train.shape) #(60000,)
print(x_test.shape)
print(t_test.shape)
