# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('maaslar.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)
#veri on isleme


x = veriler.iloc [:,1:2]
y = veriler.iloc [:,2:]

X= x.values
Y= y.values 



#verilerin egitim ve test icin bolunmesi
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x,y)

plt.scatter(X,Y, color='red')
plt.plot(x, lin_reg.predict(X), color='blue')
plt.show()

#poly

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)

print (x_poly)


lin_reg2= LinearRegression()

lin_reg2.fit(x_poly,y)

plt.scatter(x,y)
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='yellow')
plt.show()

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)

print (x_poly)


lin_reg2= LinearRegression()

lin_reg2.fit(x_poly,y)

plt.scatter(x,y)
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='green')
plt.show()



print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))








