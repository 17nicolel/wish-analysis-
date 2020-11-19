import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

pd.set_option('display.max_columns',None)
product=pd.read_csv('summer-products-with-rating-and-performance_2020-08.csv')
tag=product['tags']
title=product['title']
unitssold=product['units_sold']
tags_units=pd.concat([tag,unitssold],axis=1)

tags_seperated = pd.DataFrame(tags_units['tags'].str.split(',').values.tolist())
product2=pd.concat([tags_seperated,unitssold],axis=1)

product_dummies=pd.get_dummies(product2, columns=[1,2])
product_dummies.drop(product_dummies.iloc[:, 0:40], inplace = True, axis = 1)
soldabove1000units=tags_units.units_sold>=1000
product_dummies=pd.concat([soldabove1000units,product_dummies],axis=1)
print(product_dummies.head())

X = product_dummies.iloc[:,1:]
y = product_dummies.iloc[:,0]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2020)
#Partition the data into train and test sets (70/30). Use random_state = 2020

LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
y_pred = LogReg.predict(X_test)
#Fit the training data to a logistic regression model

print(metrics.accuracy_score(y_test, y_pred))
#Display the accuracy of your predictions for selling

confusionmatrix = metrics.confusion_matrix(y_test, y_pred)
print(confusionmatrix)
plot_confusion_matrix(LogReg, X_test, y_test)
plt.show()
#Display the confusion matrix along with the labels (Yes, No).

print(len(product_dummies.columns))

import random
randomlist = []
for i in range(0,529):
    n = random.randint(0,1)
    randomlist.append(n)
test = np.reshape(randomlist,(1,-1))
print(len(test))
result = LogReg.predict(test)
print('The item with these tags are predicted to sell above 1000 units:',result)

