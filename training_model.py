#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('train.csv')

def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'No title in name'

titles = set([x for x in dataset.Name.map(lambda x: get_title(x))])

def shorter_titles(x):
    title = x['Title']
    if title in ['Capt', 'Col', 'Major']:
        return 'Officer'
    elif title in ['Jonkheer', 'Don', 'the Countess', 'Dona', 'Lady', 'Sir']:
        return 'Royalty'
    elif title =='Mme':
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    else:
        return title
    
dataset['Title'] = dataset['Name'].map(lambda x: get_title(x))
dataset['Title'] = dataset.apply(shorter_titles, axis=1)

dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)
dataset['Embarked'].fillna('S', inplace=True)
dataset.drop('Cabin', axis=1, inplace=True)
dataset.drop('Ticket', axis=1, inplace=True)
dataset.drop('Name', axis=1, inplace=True)
dataset.Sex.replace(('male', 'female'), (0,1), inplace=True)
dataset.Embarked.replace(('S', 'C', 'Q'), (0,1,2), inplace=True)
dataset.Title.replace(('Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev', 'Royalty', 'Officer'), (0,1,2,3,4,5,6,7), inplace=True)

x = dataset.drop(['Survived','PassengerId'], axis=1)
y = dataset['Survived']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1)
randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)

pickle.dump(randomforest, open('titanic_model.sav', 'wb'))

