# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 23:01:18 2019

@author: Utkarsh Kumar
"""
#importing libraries
import pandas as pd  
import numpy as np

data = pd.read_excel('Lending Data.xlsx') #importing the data
# Descriptive analysis
data.describe()
# Getting general info about columns
data.info()
#Getting names of the columns in the data
data.columns
#Checking the null values-Returns the boolen value i.e either true or false
data.isnull()
# Checking the top 7 rows data from top
data.head(7)
# Checking the bottom 5 rows data
data.tail()
 
#Removing the first row
data = data.drop(data.index[0])
#Another way around to remove a row if certain column is null
data = data.dropna(axis=0, subset=['Name of Lending Institution'])
data = data[pd.notnull(data['Name of Lending Institution'])]

data.info()
#Replacing the - with nan
data1 = data.replace('-',np.nan)
#Replacing the nan with mean - This works only for number columns
data1 = data1.replace(np.nan,data1.mean())
data2 = data1.iloc[:-3,:]

#Removing the columns
data2 = data2.drop(['CC Amount/TA1'],axis = 1)
#Removing the columns where all null values
data2 = data2.dropna(axis = 1, how = 'all')
#Replacing 0 with null
data2 = data2.replace(0,data2.mean())

#Filling the NR values in Rank column
data2.iloc[88:93,2] = list([89,90,91,92,93])
#Finally checking is there any null values left
data2.isnull().sum()
#Checking any - left or not..
(data2=='-').sum()

#Upto here we are done with basic cleaning

# Some visualisation using boxplot
import seaborn as sns
sns.boxplot(y='Amount ($1,000)',data=data2)
sns.boxplot(y='Amount ($1,000).1',data=data2)
sns.boxplot(y='Amount ($1,000).2',data=data2)
#From graph its clear that data is not normal and contains many outliers

#Now moving towards the advanced part of data cleaning using sklearn :) 

#Dummy value creation
#Creating dummy
dummies = pd.get_dummies(data['Name of Lending Institution'])
#Concat original with dumy
data3 = pd.concat([data2,dummies],axis=1)
#Drop original column
data3 = data3.drop(['Name of Lending Institution'],axis = 1)

# importing sklearn


from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split #sklearn.cross_validation


wine = datasets.load_wine()

X = wine.data[:, :]
Y = wine.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train) #fit is used to put the data in equation of Stasndard scaler

standardized_X = scaler.transform(X_train) #transform is to get the final variable.

from sklearn.preprocessing import Normalizer

scaler = Normalizer().fit(X_train)   #It is used to put the data in equation of normaliazation

normalized_X = scaler.transform(X_train)    #transform is to get the final variable.

df = pd.read_csv('letterdata.csv')
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer() #Performs the task of dummy value creation
lb_state = lb.fit_transform(df['letter'])

from sklearn.preprocessing import LabelEncoder   #Encode the values but problem is that in this there is chance that the algorithm 
#..may understand the values as value level like 0 for least priority and bigger number as higher priority
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df['letter'])


label_encoder = LabelEncoder()
xy = label_encoder.fit(df['letter'])
standardized_X = xy.transform(df['letter'])




from sklearn.preprocessing import OneHotEncoder  #Same as dummy value creation
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

from sklearn.preprocessing import Binarizer  #Giving error could not convert string to float: 'T'  ????
binarizer = Binarizer(threshold=13.0).fit(df)
binary_X = binarizer.transform(df)

from sklearn.preprocessing import Imputer  #Imputer giving the deprecation warning..
imp = Imputer(missing_values=13, strategy='mean', axis=0)
imp.fit_transform(df)
e = imp.fit_transform(df)
