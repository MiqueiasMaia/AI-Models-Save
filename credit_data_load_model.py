# -*- coding: utf-8 -*-

import pandas as pd
import pickle
import numpy


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

base = pd.read_csv('Model-save/credit_db_discrete.csv')
previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values

imputer = SimpleImputer(missing_values=numpy.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

nb = pickle.load(open('nb_classificador.sav', 'rb'))
tree = pickle.load(open('tree_classificador.sav', 'rb'))
random_forest = pickle.load(open('randomForest_classificador.sav', 'rb'))

resultado_nb = nb.score(previsores, classe)
resultado_tree = tree.score(previsores, classe)
resultado_forest = random_forest.score(previsores, classe)

# print(resultado_nb)
# print(resultado_tree)
# print(resultado_forest)

novo = [[54000,40,5000]]
novo = numpy.asarray(novo)
novo = novo.reshape(-1,1)
novo = scaler.fit_transform(novo)
novo = novo.reshape(-1,3)

predicao_nb = nb.predict(novo)
predicao_tree = tree.predict(novo)
predicao_fores = random_forest.predict(novo)

print(predicao_nb)
print(predicao_tree)
print(predicao_fores)