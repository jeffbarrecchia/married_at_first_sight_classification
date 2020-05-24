#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 16:48:55 2020

@author: jeffbarrecchia
"""

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as ttSplit
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('~/Documents/Kaggle_Projects/marriedAtFirstSight.csv')



no_names = df.drop(columns = ['Name'])
dummies = pd.get_dummies(no_names, columns = ['Location', 'Gender', 'Occupation', 'Decision', 'Status'])

df_x = dummies.drop(columns = ['Status_Divorced', 'Status_Married'])
df_y = dummies[['Status_Divorced', 'Status_Married']]

x_train, x_test, y_train, y_test = ttSplit(df_x, df_y, train_size = 0.8, random_state = 4)

tree = DecisionTreeClassifier(max_depth = 3)
tree.fit(x_train, y_train)

print('The training accuracy is: {:.3f}'.format(tree.score(x_train, y_train) * 100) + '%')
print('The test accuracy is: {:.3f}'.format(tree.score(x_test, y_test) * 100) + '%')

tree_predict = tree.predict(df_x)

tree_predicted = []

for i in tree_predict:
    tree_predicted.append(i[1:])
    
df['Decision Tree Predicted'] = tree_predicted


forest = RandomForestClassifier(max_depth = 5)
forest.fit(x_train, y_train)

print('\nThe training accuracy is: {:.3f}'.format(forest.score(x_train, y_train) * 100) + '%')
print('The test accuracy is: {:.3f}'.format(forest.score(x_test, y_test) * 100) + '%')

forest_predict = forest.predict(df_x)

forest_predicted = []

for i in forest_predict:
    forest_predicted.append(i[1:])
    
df['Random Forest Predicted'] = forest_predicted

df.to_csv('~/Documents/DS_Projects/predictionDataframe.csv')

length = range(0, 68)
tree_acc = tree.score(x_test, y_test)
forest_acc = forest.score(x_test, y_test)

knc = KNeighborsClassifier(n_neighbors = 54)
knc.fit(x_train, y_train)

print('\nKNeighbors training accuracy was: {:.3f}'.format(knc.score(x_train, y_train) * 100) + '%')
print('KNeighbors test accuracy was: {:.3f}'.format(knc.score(x_test, y_test) * 100) + '%')    

# plt.plot(neighbor_settings, neighbor_training_acc, label = 'Training Accuracy')
# plt.plot(neighbor_settings, neighbor_test_acc, label = 'Test Accuracy')
# plt.xlabel('n_neighbors')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# sb.countplot(no_names['Gender'])
# sb.countplot(no_names['Decision'], hue = no_names['Status'])
# sb.countplot(no_names['Status'])
# sb.distplot(no_names['Age'])
# plt.ylabel('Frequency')
# plt.title('Distribution of Age')
# sb.swarmplot(no_names['Status'], no_names['Age'])
# sb.swarmplot(no_names['Decision'], no_names['Age'])
# plt.xlabel('Stay Together After Eight Weeks?')













