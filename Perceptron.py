""" 

*기초 퍼셉트론*

def AND(x1, x2): # 입력 신호
    w1, w2, theta = 0.5, 0.5, 0.7 #가중치 및 임계값 설정
    tmp = x1*w1 + x2*w2
    
    if tmp >= theta:
        return 0
    elif tmp < theta:
        return 1
    
    print(theta)

"""
"""
배열로 퍼셉트론 간단하게 표현
import numpy as np

x = np.array([0, 1]) # 입력
w = np.array([0.5, 0.5]) # 가중치
b = -0.7 # 편향

np.sum(w*x) + b # 퍼셉트론

*배열 활용 퍼셉트론*

# 퍼셉트론의 AND, NAND, OR은 가중치와 편향의 차이. 나머지는 일치
import numpy as np

def AND(x1, x2):

    x = np.array([x1, x2]) # 입력
    w = np.array([0.5, 0.5]) # 가중치
    b = -0.7 # 편향 (연산 후 편향보다 높은 값일 때 1 반환)
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(x*w)+b
    
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(x*w) + b
    
    if tmp <= 0:
        return 0
    else:
        return 1

#다층 퍼셉트론(XOR)

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

***행렬***
A = np.array([1, 2, 3, 4])
print(A)

np.ndim(A)

A.shape
A.shape(4)

B = np.array([[1, 2], [3, 4], [5, 6]])

print(B)

np.ndim(B)

B.shape


A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[1, 3], [5, 7], [2, 4]])

D = np.dot(A, B) #행렬 곱

print(D) #ValueError: shapes (3,2) and (3,2) not aligned: 2 (dim 1) != 3 (dim 0)

A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[1, 3, 5], [5, 7, 9]])

D = np.dot(A, B) #행렬 곱

print(D) 

#결과
#[[11 17 23]
#[23 37 51]
#[35 57 79]]

#신경망의 계산은 행렬의 곱으로 수행. 이때, 차원의 원소수를 고려해야 함. 
#1층에서 뻗어나가는 basis의 수가 받는 2층 원소의 수와 일치해야 함
#즉, X -> Y에서 X의 수와 무관하게 W = Y

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


#3층 신경망 구현 기초

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1

Z1 = sigmoid(A1)

print("A1:", A1)
print("Z1:", Z1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

def identify_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identify_function(A3) # 혹은 Y = A3

***3층 퍼셉트론론 구현***

import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identify_function(x):
    return x

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identify_function(a3)
    
    return y

network = init_network()

x = np.array([1.0, 0.5])
y = forward(network, x)

print(y)
    
***소프트맥스 함수***
a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a) # 지수 함수
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)

def softmax(a): #소프트맥스 함수는 지수함수로 인한 거대수 문제 발생 가능. 이때, 나눗셈 계산시 수치 불안정
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y
     
a = np.array([1010, 1000, 990])
s1 = np.exp(a) / np.sum(np.exp(a))
print(s1) #[nan nan nan], 계산 불가

c = np.max(a) # c = 1010 (배열 중 최대값)
s2 = np.exp(a - c) / np.sum(np.exp(a - c))
print(s2) # [9.99954600e-01 4.53978686e-05 2.06106005e-09], 이처럼 입력 신호 중 최대값을 빼주면 계산 가능
           
"""

import numpy as np

#위의 내용을 적용한 소프트맥스 함수 구현

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)

print(y) # [0.01821127 0.24519181 0.73659691]
print("sum y: ", np.sum(y)) # sum y:  1.0, 이처럼 소프트맥스 함수는 sum = 1. 덕분에 확률적 해석 가능



