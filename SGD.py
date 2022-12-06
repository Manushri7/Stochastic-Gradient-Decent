#!/usr/bin/env python
# coding: utf-8

# # Boston House price prediction using SGD

# In this kernel we will be implementing SGD on LinearRegression from scarch using python and we will be also comparing sklearn implementation SGD and our implemented SGD.

# In[38]:



import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
boston = load_boston()


# In[39]:


#REFERENCES


#1)https://www.kaggle.com/premvardhan/stocasticgradientdescent-implementation-lr-python
#2)https://medium.com/@nikhilparmar9/simple-sgd-implementation-in-python-for-linear-regression-on-boston-housing-data-f63fcaaecfb1


# In[2]:


print(boston.data.shape)


# In[3]:


print(boston.feature_names)


# In[4]:


print(boston.target.shape)


# In[5]:


print(boston.DESCR)


# In[6]:


# Loading data into pandas dataframe
bos = pd.DataFrame(boston.data)
print(bos.head())


# In[7]:


bos['PRICE'] = boston.target

X = bos.drop('PRICE', axis = 1)
Y = bos['PRICE']


# In[8]:


# Split data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[9]:


X_train.mean()


# In[10]:


# Standardization

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.fit_transform(X_test)


# In[11]:


X_train


# In[12]:


from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
clf = SGDRegressor()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

print("Coefficients: \n", clf.coef_)
print("Y_intercept", clf.intercept_)


# **Observations**
# 
# * Overall we can say the regression line not fits data perfectly but it is okay. But our goal is to find the line/plane that best fits our data means minimize the error i.e. mse should be close to 0.
# * MSE is 28.54 means the total loss(squared difference of true/actual target value and predicted target value). 0.0 is perfect i.e. no loss.
# * coefficient of determination tells about the goodness of fit of a model and here, r^2 is 0.70 which means regression prediction does not perfectly fit the data. An r^2 of 1 indicates that regression prediction perfect fit the data.

# # Stochastic Gradient Decent(SGD) for Linear Regression

# In[13]:


# Imported necessary libraries
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# In[14]:


# Data loaded 
bostan = load_boston()


# In[15]:


# Data shape
bostan.data.shape


# In[16]:


# Feature name
bostan.feature_names


# In[17]:


# This is y value i.e. target
bostan.target.shape


# In[18]:


# Convert it into pandas dataframe
data = pd.DataFrame(bostan.data, columns = bostan.feature_names)
data.head()


# In[19]:


# Statistical summary
data.describe()


# In[20]:


#noramlization for fast convergence to minima
data = (data - data.mean())/data.std()
data.head()


# In[21]:


data.mean()


# In[23]:


# MEDV(median value is usually target), change it to price
data["PRICE"] = bostan.target
data.head()


# In[24]:


# Target and features
Y = data["PRICE"]
X = data.drop("PRICE", axis = 1)


# In[25]:



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[26]:


x_train.insert(x_train.shape[1],"PRICE",y_train)


# In[27]:


def cost_function(b, m, features, target):
    totalError = 0
    for i in range(0, len(features)):
        x = features
        y = target
        totalError += (y[:,i] - (np.dot(x[i] , m) + b)) ** 2
    return totalError / len(x)


# In[28]:



def r_sq_score(b, m, features, target):
    for i in range(0, len(features)):
        x = features
        y = target
        mean_y = np.mean(y)
        ss_tot = sum((y[:,i] - mean_y) ** 2)
        ss_res = sum(((y[:,i]) - (np.dot(x[i], m) + b)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
    return r2


# In[29]:


def gradient_decent(w0, b0, train_data, x_test, y_test, learning_rate):
    n_iter = 500
    partial_deriv_m = 0
    partial_deriv_b = 0
    cost_train = []
    cost_test = []
    for j in range(1, n_iter):
        
        # Train sample
        train_sample = train_data.sample(160)
        y = np.asmatrix(train_sample["PRICE"])
        x = np.asmatrix(train_sample.drop("PRICE", axis = 1))
        
        for i in range(len(x)):
            partial_deriv_m += np.dot(-2*x[i].T , (y[:,i] - np.dot(x[i] , w0) + b0))
            partial_deriv_b += -2*(y[:,i] - (np.dot(x[i] , w0) + b0))
        
        w1 = w0 - learning_rate * partial_deriv_m 
        b1 = b0 - learning_rate * partial_deriv_b
        
        if (w0==w1).all():
            
            break
        else:
            w0 = w1
            b0 = b1
            learning_rate = learning_rate/2
       
            
        error_train = cost_function(b0, w0, x, y)
        cost_train.append(error_train)
        error_test = cost_function(b0, w0, np.asmatrix(x_test), np.asmatrix(y_test))
        cost_test.append(error_test)
        
       
        
    return w0, b0, cost_train, cost_test


# In[30]:



learning_rate = 0.001
w0_random = np.random.rand(13)
w0 = np.asmatrix(w0_random).T
b0 = np.random.rand()

optimal_w, optimal_b, cost_train, cost_test = gradient_decent(w0, b0, x_train, x_test, y_test, learning_rate)
print("Coefficient: {} \n y_intercept: {}".format(optimal_w, optimal_b))

'''
error = cost_function(optimal_b, optimal_w, np.asmatrix(x_test), np.asmatrix(y_test))
print("Mean squared error:",error)
'''

plt.figure()
plt.plot(range(len(cost_train)), np.reshape(cost_train,[len(cost_train), 1]), label = "Train Cost")
plt.plot(range(len(cost_test)), np.reshape(cost_test, [len(cost_test), 1]), label = "Test Cost")
plt.title("Cost/loss per iteration")
plt.xlabel("Number of iterations")
plt.ylabel("Cost/Loss")
plt.legend()
plt.show()


# ### observations

# 1)as per number of iterations there is no change in error rate

# # Comparison between sklearn SGD and implemented SGD in python 

# In[31]:



print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))

print("Variance score: %.2f" % r2_score(Y_test, Y_pred))


# In[32]:



error = cost_function(optimal_b, optimal_w, np.asmatrix(x_test), np.asmatrix(y_test))
print("Mean squared error: %.2f" % (error))

r_squared = r_sq_score(optimal_b, optimal_w, np.asmatrix(x_test), np.asmatrix(y_test))
print("Variance score: %.2f" % r_squared)


# In[33]:



plt.figure(1)
plt.subplot(211)
plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: Sklearn SGD")
plt.show()

# Implemented SGD
plt.subplot(212)
plt.scatter([y_test], [(np.dot(np.asmatrix(x_test), optimal_w) + optimal_b)])
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: Implemented SGD")
plt.show()


# ### observations

# 1) Custom SGD and  sklearn inbuild SGD almost gives same result

# In[34]:


# Distribution of error
delta_y_im = np.asmatrix(y_test) - (np.dot(np.asmatrix(x_test), optimal_w) + optimal_b)
delta_y_sk = Y_test - Y_pred
import seaborn as sns;
import numpy as np;
sns.set_style('whitegrid')
sns.kdeplot(np.asarray(delta_y_im)[0], label = "Implemented SGD")
sns.kdeplot(np.array(delta_y_sk), label = "Sklearn SGD")
plt.title("Distribution of error: $y_i$ - $\hat{y}_i$")
plt.xlabel("Error")
plt.ylabel("Density")
plt.legend()
plt.show()


# ### observation

# 1) Implemented SGD gives positive side of error more than negative side errors

# In[35]:


# Distribution of predicted value
sns.set_style('whitegrid')
sns.kdeplot(np.array(np.dot(np.asmatrix(x_test), optimal_w) + optimal_b).T[0], label = "Implemented SGD")
sns.kdeplot(Y_pred, label = "Sklearn SGD")
plt.title("Distribution of prediction $\hat{y}_i$")
plt.xlabel("predicted values")
plt.ylabel("Density")
plt.show()


# **observations**
# 1) The mean squared error(mse) is quite high means the regression line does not fit the data properly. i.e. average squared difference between the actual target value and predicted target value is high. lower value is better.
# 
# 2) After looking at the error graph we can say +ve side of the graph, error is more.
# 

# **Conclusions**
# * While comparing scikit-learn implemented linear regression and explicitly implemented linear regression using optimization algorithm(sgd) in python we see there are not much differences between both of them.
# * Both of the model are not perfect but okay.
