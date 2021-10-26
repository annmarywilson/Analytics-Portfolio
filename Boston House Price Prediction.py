#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


# # Importing the datasets

# In[3]:


house_price_dataset = sklearn.datasets.load_boston()


# In[4]:


print(house_price_dataset)


# # Loading dataset to pandas dataframe

# In[8]:


house_price_df = pd.DataFrame(house_price_dataset.data, columns = house_price_dataset.feature_names)


# In[9]:


#Print first 5 rows of the dataframe
house_price_df.head(5)


# In[10]:


# Adding the target array as price to the dataframe
house_price_df['PRICE'] = house_price_dataset.target


# In[11]:


house_price_df.head()


# In[13]:


#Checking the number of rows and columns in the dataframe
house_price_df.shape


# In[15]:


#Checking for missing values
house_price_df.isnull().sum()


# In[16]:


#Gathering the statistical data frame
house_price_df.describe()


# # Finding the correleation between the features in the data frame

# In[18]:


correlation = house_price_df.corr()


# In[28]:


#Creating heatmap to find the correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation, annot=True, fmt=".1f", annot_kws={'size':8}, square=True)


# # Splitting the data frame

# In[30]:


X = house_price_df.drop(['PRICE'], axis=1)
Y = house_price_df['PRICE']


# In[36]:


print(X)
print(Y)


# 
# # Splitting the data as Training Data and Test Data

# In[42]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)


# In[43]:


print(X.shape, X_train.shape, X_test.shape)


# # Model training
#    XGBoost Regresor

# In[45]:


#Loading the model
model = XGBRegressor()


# In[47]:


#Training the model with X_train
model.fit(X_train, Y_train)


# # Evaluating
# Predicting on train data

# In[50]:


#Accuracy for predicting on train data
model_train_predict = model.predict(X_train)
print(model_train_predict)


# In[52]:


#R Squared Error
r2_score = metrics.r2_score(Y_train, model_train_predict)
print(r2_score)


# In[54]:


#Mean Absolute Error
mae_score = metrics.mean_absolute_error(Y_train, model_train_predict)
print(mae_score)


# # Visualizing Actual Vs Predicted Price

# In[60]:


plt.scatter(Y_train, model_train_predict)
plt.xlabel('Actual price')
plt.ylabel('Predicted price')
plt.title('Actual Price Vs Predicted Price')
plt.show()


# # Prediction on test data

# In[56]:


#Accuracy for predicting on train data
model_test_predict = model.predict(X_test)
print(model_test_predict)
#R Squared Error
r2_score = metrics.r2_score(Y_test, model_test_predict)
print(r2_score)
#Mean Absolute Error
mae_score = metrics.mean_absolute_error(Y_test, model_test_predict)
print(mae_score)


# In[ ]:




