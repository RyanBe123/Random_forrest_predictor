# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 09:51:27 2021

@author: ryanb
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\ryanb\Documents\Careers\QuantSpark\HR_comma_sep.csv')



for i,item in enumerate(dataset.iloc[:,8].values):
    if item == 'low':
        dataset.iloc[i,8] = 1
    elif item == 'medium':
        dataset.iloc[i,8] = 2
    else:
        dataset.iloc[i,8] = 3
      
for i,item in enumerate(dataset.iloc[:,7].values):
    if item == 'accounting':
        dataset.iloc[i,7] = 1
    elif item == 'hr':
        dataset.iloc[i,7] = 2
    elif item == 'IT':
        dataset.iloc[i,7] = 3
    elif item == 'management':
        dataset.iloc[i,7] = 4
    elif item == 'marketing':
        dataset.iloc[i,7] = 5
    elif item == 'product_mng':
        dataset.iloc[i,7] = 6
    elif item == 'RandD':
        dataset.iloc[i,7] = 7
    elif item == 'sales':
        dataset.iloc[i,7] = 8
    elif item == 'support':
        dataset.iloc[i,7] = 9
    elif item == 'technical':
        dataset.iloc[i,7] = 10

print(dataset.head())

x = dataset.iloc[:,0:9].values
y = dataset.iloc[:,9].values


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

regressor = RandomForestRegressor(n_estimators= 100, random_state = 0)
regressor.fit(x_train,y_train)    
y_pred = regressor.predict(x_test)

importances = list(regressor.feature_importances_)
Variables = list(dataset.columns)
Variables = Variables[:-1]

labels = ['satisfaction \n level','last\n evaluation', 'number \n projects',	
          'average\n monthly \n hours',	'time\n spent \n company',	'Work \n accidents',	
          'promotion \n last \n 5years','Department','salary']

plt.bar(Variables,importances, orientation = 'vertical',color=['black', 'red', 'green', 'blue', 'cyan', 'yellow','orange','pink','purple'])
plt.xlabel('Variables')
plt.ylabel('Relative Strength of Dependant Variable')
#plt.xticks(fontsize = 8, rotation = 45)
plt.xticks(np.arange(9), labels = labels,fontsize = 8, rotation = 90 )
#plt.savefig(r'C:\Users\ryanb\Documents\Careers\QuantSpark\importances')
plt.show()

#plt.rcParams.update({'font.size': 4}) doesnt work


ypred_2 = np.zeros(3000)
for i in np.arange(0,len(y_pred)):
    if y_pred[i] < 0.5:  #setting those scores less than 0.5 as a person who would leave
        ypred_2[i] = 0
    else:
        ypred_2[i]= 1

print('Accuracy',metrics.accuracy_score(y_test,ypred_2))


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
#print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



random_employee = np.array([0.85,0.76,3,150,3,0,2,2,2])
random_employee = random_employee.reshape(1,-1)
random_test = regressor.predict(random_employee)
print(random_test) #left =1, stay = 0
#print('Mean Absolute Error:', metrics.mean_absolute_error(random_test, y_pred))
#isolating high performers and determining their attributes



        