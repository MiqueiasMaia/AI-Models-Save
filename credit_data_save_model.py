# -*- coding: utf-8 -*-

import pandas
import numpy
base = pandas.read_csv('Model-save/credit_db_discrete.csv')

previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=numpy.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.naive_bayes import GaussianNB #ok
from sklearn.tree import DecisionTreeClassifier #ok
from sklearn.ensemble import RandomForestClassifier #ok

classificadorNB = GaussianNB()
classificadorNB.fit(previsores, classe)

classificadorTree = DecisionTreeClassifier()
classificadorTree.fit(previsores, classe)

classificadorRandomForest = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
classificadorRandomForest.fit(previsores, classe)

import pickle
pickle.dump(classificadorNB, open('nb_classificador.sav','wb'))
pickle.dump(classificadorTree, open('tree_classificador.sav','wb'))
pickle.dump(classificadorRandomForest, open('randomForest_classificador.sav','wb'))