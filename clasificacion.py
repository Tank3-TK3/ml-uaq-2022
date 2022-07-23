# -*- coding: utf-8 -*-
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("http://virtualfif.uaq.mx/diplomado/data/clasificacion/Social_Network_Ads.csv")

#print(df.head())
#print(df.info())
#print(df.columns)

'''plt.scatter(df['Gender'], df['Purchased'])
plt.xlabel('Gender')
plt.ylabel('Purchased')
plt.grid()
plt.show()'''

grupo1 = df.groupby(['Purchased', 'Gender']).count()['User ID']
grupo1.unstack(level=0).plot.bar()
plt.show()

grupo2 = df.groupby(['Purchased', 'Age']).count()['User ID']
grupo2.unstack(level=0).plot.bar()
plt.show()

grupo3 = df.groupby(['Purchased', 'EstimatedSalary']).count()['User ID']
grupo3.unstack(level=0).plot.bar()
plt.show()

X = df.iloc[:,[2,3]].values
y = df.iloc[:,4]

sc_X = StandardScaler()
X = sc_X.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)

log = LogisticRegression(random_state=0)
log.fit(X_train, y_train)
print(f'Train (log): {log.score(X_train, y_train)}')
print(f' Test (log): {log.score(X_test, y_test)}')
y_pred_log = log.predict(X_test)
cm_log = confusion_matrix(y_test, y_pred_log)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(f'Train (knn): {knn.score(X_train, y_train)}')
print(f' Test (knn): {knn.score(X_test, y_test)}')
y_pred_knn = knn.predict(X_test)
cm_knn = confusion_matrix(y_test, y_pred_knn)

arbol = DecisionTreeClassifier(criterion='entropy', random_state=0)
arbol.fit(X_train, y_train)
print(f'Train (arbol): {arbol.score(X_train, y_train)}')
print(f' Test (arbol): {arbol.score(X_test, y_test)}')
y_pred_arbol = arbol.predict(X_test)
cm_arbol = confusion_matrix(y_test, y_pred_arbol)

bosque = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=0)
bosque.fit(X_train, y_train)
print(f'Train (bosque): {bosque.score(X_train, y_train)}')
print(f' Test (bosque): {bosque.score(X_test, y_test)}')
y_pred_bosque = bosque.predict(X_test)
cm_bosque = confusion_matrix(y_test, y_pred_bosque)