# -*- coding: utf-8 -*-
"""parkinsons.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZNZMwnRyQ56lm6gcy-BUhunNgtr3Z505
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

data=pd.read_csv('/content/parkinsons.csv')

data.shape

data.head()

data.info()

data.describe()

data['status'].value_counts()

#grouping data bases on status column
data.drop(columns='name',axis=1).groupby('status').mean()

"""data preprocessing"""

X=data.drop(columns=['name','status'],axis=1)
Y=data['status']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)

"""Data standardization"""

scaler=StandardScaler()
X=scaler.fit(X_train)

X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
print(X_train)

"""model training

support vector machine model(SVM)
"""

model=svm.SVC(kernel='linear')

model.fit(X_train,Y_train)

"""ModelEvaluation"""

pred_tr=model.predict(X_train)
tr_acc=accuracy_score(pred_tr,Y_train)

pred_te=model.predict(X_test)
te_acc=accuracy_score(pred_te,Y_test)

print('training accuracy : ',tr_acc,'\ntest accuracy : ',te_acc)

"""Building predictive system"""

input=(119.99200,157.30200,74.99700,0.00784,0.00007,0.00370,0.00554,0.01109,0.04374,0.42600,0.02182,0.03130,0.02971,0.06545,0.02211,21.03300,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654)
input_np=np.asarray(input)
input_reshaped=input_np.reshape(1,-1)
std_in=scaler.transform(input_reshaped)
prediction=model.predict(std_in)
if prediction[0]==1:
  print('there is high chance that you have parkinson\'s!!!!')
else:
  print('you do not have parkinson\'s')

input=(198.38300,215.20300,193.10400,0.00212,0.00001,0.00113,0.00135,0.00339,0.01263,0.11100,0.00640,0.00825,0.00951,0.01919,0.00119,30.77500,0.465946,0.738703,-7.067931,0.175181,1.512275,0.096320)
input_np=np.asarray(input)
input_reshaped=input_np.reshape(1,-1)
std_in=scaler.transform(input_reshaped)
prediction=model.predict(std_in)
if prediction[0]==1:
  print('there is high chance that you have parkinson\'s!!!!')
else:
  print('you do not have parkinson\'s')

