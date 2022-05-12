# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('odev_tenis.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)
#veri on isleme




'''

play = veriler.iloc[:,4:5].values
print(play)

windy= veriler.iloc[:,3:4].values

print(windy)



le= preprocessing.LabelEncoder()

windy[:,0]= le.fit_transform(windy)

print(windy)

ohe= preprocessing.OneHotEncoder()

windy= ohe.fit_transform(windy).toarray()

print(windy)


outlook= veriler.iloc[:,0:1].values

print(outlook)

outlook= ohe.fit_transform(outlook).toarray()



play= ohe.fit_transform(play).toarray()

'''

veriler2 = veriler.apply(LabelEncoder().fit_transform)


ohe= preprocessing.OneHotEncoder()

outlook= veriler.iloc[:,0:1].values

outlook= ohe.fit_transform(outlook).toarray()



#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data=outlook, index = range(14), columns = ['overcast','rainy','sunny'])
print(sonuc)


sonuc3= pd.DataFrame(data=veriler.iloc[:,1:3].values, index= range(14), columns=['temperature','humidity'])



#dataframe birlestirme islemi
s=pd.concat([veriler2.iloc[:,-2:],sonuc,sonuc3], axis=1)
print(s)




#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s.iloc[:,:-1],s.iloc[:,-1:],test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)

y_prod= regressor.predict(x_test)





X= np.append(arr= np.ones((14,1)).astype(int), values= s.iloc[:,:-1], axis=1)

X_1 = s.iloc[:, [0,1,2,3,4,5,6]].values
print(X_1)


X_1 = np.array(X_1, dtype= float)
model = sm.OLS (s.iloc[:,-1:], X_1).fit()


print (model.summary())

'''
X_1 = veri.iloc[:, [0,1,2,3,5]].values
X_1 = np.array(X_1, dtype= float)
model = sm.OLS (boy, X_1).fit()


print (model.summary())


X_1 = veri.iloc[:, [0,1,2,3]].values
X_1 = np.array(X_1, dtype= float)
model = sm.OLS (boy, X_1).fit()


print (model.summary())

'''