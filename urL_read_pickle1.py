#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 00:25:12 2022

@author: semihdalgin
"""

import pandas as pd


url= "http://bilkav.com/satislar.csv"

veriler = pd.read_csv(url)

veriler= veriler.values

X= veriler[:,0].reshape(-1,1)
Y= veriler[:,1].reshape(-1,1)


bolme= 0.33



from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y, test_size=bolme)




import pickle as pc

dosya="model.kayit"

yuklenen = pc.load(open(dosya, 'rb'))

print(yuklenen.predict(X_test))
