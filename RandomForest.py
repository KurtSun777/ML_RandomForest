#!/usr/bin/env python
# coding: utf-8

# # RandomForest

# In[1]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz 
from xgboost import plot_importance
import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('real_309_4_en.csv')
df.head(3)


# In[3]:


X = df.drop(['label', 'years', 'law'],axis=1)
y = df['label']


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(max_depth=6, n_estimators=10, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
num_correct_samples = accuracy_score(y_test, y_pred, normalize=False)
con_matrix = confusion_matrix(y_test, y_pred)

print('number of correct sample: ', num_correct_samples)
print('accuracy: ', accuracy)
print('Precision:', precision_score(y_test, y_pred, average = 'weighted'))
print('Recall:', recall_score(y_test, y_pred, average = 'weighted'))
print('F1:', f1_score(y_test, y_pred, average = 'weighted'))
print('confusion matrix: ', con_matrix)


# In[ ]:




