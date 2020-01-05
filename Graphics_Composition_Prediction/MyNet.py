
# coding: utf-8

# In[1]:

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v3 import *

np.random.seed(1)


# In[2]:

X_train, Y_train, X_test, Y_test, classes = load_data()
X_train = X_train.reshape(X_train.shape[0],-1).T
X_test = X_test.reshape(X_test.shape[0],-1).T
Y_train = (Y_train).T
Y_test = (Y_test).T


# In[3]:

def init_params(layers):
    params={}
    L = len(layers)
    for l in range(1,L):
        params['W'+str(l)] = np.random.randn(layers[l],layers[l-1])*0.01
        params['b' + str(l)] = np.zeros((layers[l],1))
        
    return params


# In[4]:

layers=(X_train.shape[0],16,8,1)
params = init_params(layers)


# In[5]:

def sigmoid(x):
    z = 1/(1+np.exp(-x))
    return z


# In[6]:

def tanh(x):
    z = (np.exp(x) - np.exp(-x)) / (np.exp(x)+np.exp(-x))
    return z
    


# In[7]:

def RELU(x):
    return np.maximum(x,0.01*x)


# In[8]:

def NN(X_train,Y_train,params,iteration,alpha):
    
    m = X_train.shape[1]
    lambd = 0
    L = len(layers)
    for l in range(1,L):
        params['W'+str(l)] = np.random.randn(layers[l],layers[l-1])*0.01
        params['b' + str(l)] = np.zeros((layers[l],1))
    Z = {}
    A = {}
    dZ = {}
    dW = {}
    db = {}
    A['A' + str(0)] = X_train
    Z['Z' + str(0)] = X_train
    Y_train = Y_train.T
    for i in range(1,iteration):
        for l in range(1,L-1):
            Z['Z'+str(l)] = np.dot(params['W'+str(l)],A['A'+str(l-1)]) + params['b'+str(l)]
            A['A'+str(l)] = RELU(Z['Z'+str(l)])
            
        Z['Z'+str(L-1)] = np.dot(params['W'+str(L-1)],A['A'+str(L-2)]) + params['b'+str(L-1)]
        A['A'+str(L-1)] = sigmoid(Z['Z'+str(L-1)])
        A_pred = A['A'+str(L-1)]
        cost = np.multiply(Y_train,np.log(A_pred)) + np.multiply(1-Y_train,np.log(1-A_pred))
        cost = np.sum(cost,axis=1,keepdims=True)*(-1/m)
        for k in range(1,L):
            temp = 0
            temp = temp + np.sum(np.sum(np.dot(params['W'+str(k)],params['W'+str(k)].T)))
        cost = cost + (lambd*temp)/(2*m)
        cost = float(np.squeeze(cost))
        if i%200 ==0:
            print(cost)
        dZ['dZ' + str(L-1)] = A['A'+str(L-1)] - Y_train
        dW['dW'+str(L-1)] = np.dot(dZ['dZ' + str(L-1)],A['A'+str(L-2)].T)/m
        db['db'+str(L-1)] = np.sum(dZ['dZ' + str(L-1)], axis=1, keepdims=True)/m
        
        for l in reversed(range(1,L-1)):
            diff_temp = np.ones_like(Z['Z'+str(l)])
            diff_temp[Z['Z'+str(l)] < 0] = 0.01
            # derivative of leaky relu
            dZ['dZ'+str(l)] = np.multiply(np.dot(params['W'+str(l+1)].T, dZ['dZ'+str(l+1)]),diff_temp)
            dW['dW'+str(l)] = np.dot(dZ['dZ' + str(l)],A['A'+str(l-1)].T)/m
            db['db'+str(l)] = np.sum(dZ['dZ' + str(l)], axis=1, keepdims=True)/m
            params['W'+str(l)] = params['W'+str(l)]*(1 - (alpha*lambd)/m) - (alpha * dW['dW'+str(l)])
            params['b'+str(l)] = params['b'+str(l)] - (alpha * db['db'+str(l)])
        
    return params
    


# In[9]:

params_final = NN(X_train,Y_train,params,iteration=2001,alpha=0.001)


# In[10]:

def predict(params_final,X,Y):
    L = len(layers)
    m = X.shape[1]
    A_final = {}
    Z_final = {}
    A_final['A'+str(0)] = X
    for l in range(1,L-1):
            Z_final['Z'+str(l)] = np.dot(params_final['W'+str(l)],A_final['A'+str(l-1)]) + params_final['b'+str(l)]
            A_final['A'+str(l)] = RELU(Z_final['Z'+str(l)])
    
    Z_final['Z'+str(L-1)] = np.dot(params_final['W'+str(L-1)],A_final['A'+str(L-2)]) + params_final['b'+str(L-1)]
    x = A_final['A'+str(L-1)] = sigmoid(Z_final['Z'+str(L-1)])
    # print(x)
    x = A_final['A'+str(L-1)] = (A_final['A'+str(L-1)] >= 0.5)
    
    total = np.dot(Y,A_final['A'+str(L-1)].T) + np.dot(1-Y,(1-A_final['A'+str(L-1)]).T)
    acc = (total/m)*100
    acc = float(np.squeeze(acc))
    return acc
    


# In[11]:


train_acc =predict(params_final,X_train,Y_train.T)
test_acc = predict(params_final,X_test,Y_test.T)
print(train_acc)
print(test_acc)



