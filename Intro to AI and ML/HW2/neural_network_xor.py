
# coding: utf-8

# # Neural Network

# Implement the back-propagation algorithm to learn the weights of a perceptron with 2 input nodes, 2 hidden nodes and 1 output node.

# ## XOR

# Note: **Python3** in used.

# In[1]:


# imports
import numpy as np


# In[2]:


# parameters - input all parameter values here
# note: Unless a cell is found with changed values, the training cells the parameters here 
input_dim = 2
hidden_dim = 2 # dimensions of hidden layers
std = 0.005  # train data noise standard deviation
w_std = 0.5
learn_rate = 0.005


# In[3]:


# prepare training data
x_inputs = np.array([np.zeros(2), np.ones(2), np.array([1,0]), np.array([0,1])])
def generate_trainset(N):
    X = np.repeat(x_inputs, N//4, axis=0)
    y_xor = np.logical_xor(X.T[0], X.T[1]).astype(np.float)
    # add noise to data
    X += np.random.normal(0, std, X.shape)
    y_xor += np.random.normal(0, std, N)
    # shuffle the training data
    indices = np.arange(N)
    np.random.shuffle(indices)
    x_train, y_train = X[indices], y_xor[indices]
    return x_train, y_train


# In[4]:


def sigmoid( t):
    return 1/(1  + np.exp(-t))

def dsigmoid( t):
    return sigmoid(t)*(1 - sigmoid(t))



N = 1000
x_train, y_train = generate_trainset(N)


# In[6]:


# initialize weights
A  = np.random.normal(0, w_std, (hidden_dim, input_dim))# [1,1], [1,0], [0,0], [0,1]
a0 = np.random.normal(0, w_std, hidden_dim)
b0 = np.random.normal(0, w_std, 1)
B  = np.random.normal(0, w_std, hidden_dim)
epochs = 500 # number of itrations
for epoch in range(epochs):
    dSSE_a, dSSE_b, z_bias, y_bias = np.zeros_like(A), np.zeros_like(B), np.zeros_like(B), 0
    loss = 0
    for i, x in enumerate(x_train):
        z = sigmoid(np.dot(A,x)+a0)
        y_hat = sigmoid(np.dot(B,z)+b0)
        y_error = y_hat - y_train[i]
        y_delta = 2* y_error * dsigmoid(np.dot(B, z) + b0)
        s = dsigmoid(np.dot(A,x) + a0) * B * y_delta
        # print(s.shape)
        dSSE_b += y_delta*z
        dSSE_a += np.tensordot(s,x, axes=0)
        # print(dSSE_a.shape)
        y_bias += y_delta
        z_bias += s
        loss += y_error**2

    A  = A - learn_rate * dSSE_a
    B  = B - learn_rate * dSSE_b
    a0 = a0 - learn_rate * s
    b0 = b0 - learn_rate * y_delta

    print('Epoch: ', str(epoch+1) + '/'+str(epochs), ' Loss: ', loss/N)   


# In[7]:


def predict(x_test):
    results =  [sigmoid(np.dot(B, sigmoid(np.dot(A, x)+a0)) + b0) for x in x_test]
    return np.array(results)
def decision(x_test):
    return (predict(x_test) > 0.5).astype(int)
print(predict(x_inputs))
print(decision(x_inputs))


# ###### Experiment with N = 100

