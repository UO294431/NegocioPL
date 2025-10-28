import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_excel("Churn_Modelling_NANs.xlsx", na_values='NA')

# Seguir desde aquí
# Apartado 1
df.dropna(axis=0, how='any', inplace=True)
print(df)

# Apartado 2
COLS_X = ['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember']
X = df[COLS_X]
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Apartado 3
COL_Y = "EstimatedSalary"
Y = df[COL_Y]

# Apartado 4
lin = LinearRegression()
lin.fit(X,Y)
scores = cross_val_score(lin, X, Y, scoring='neg_mean_squared_error',cv=10)
score = scores.mean()
predicted = cross_val_predict(lin, X, Y,cv=10)
mse = mean_squared_error(Y,predicted)
r2 = r2_score(Y,predicted)
print("LIN MSE=",mse)
print("LIN score=",-score)
print("LIN R2=",r2)

svr = SVR(kernel='linear',C=1e1,epsilon=1)
svr.fit(X,Y)
scores = cross_val_score(svr, X, Y, scoring='neg_mean_squared_error',cv=10)
score = scores.mean()
predicted = cross_val_predict(svr, X, Y,cv=10)
mse = mean_squared_error(Y,predicted)
r2 = r2_score(Y,predicted)
print("SVR MSE=",mse)
print("SVR score=",-score)
print("SVR R2=",r2)

rfr = RandomForestRegressor()
rfr.fit(X,Y)
scores = cross_val_score(rfr, X, Y, scoring='neg_mean_squared_error',cv=10)
score = scores.mean()
predicted = cross_val_predict(rfr, X, Y,cv=10)
mse = mean_squared_error(Y,predicted)
r2
print("RFR MSE=",mse)
print("RFR score=",-score)
print("RFR R2=",r2)


# Apartado 5
COLS_XC = ['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember']
XC = df[COLS_XC]
Y = df['Exited']

XC = scaler.fit_transform(XC)
rfr = RandomForestClassifier()
rfr.fit(XC,Y)
scores = cross_val_score(rfr, XC, Y, scoring='accuracy',cv=10)
score = scores.mean()
predicted = cross_val_predict(rfr, XC, Y,cv=10)
acc = accuracy_score(Y,predicted)
f1 = f1_score(Y,predicted)
print("RFR ACC=",acc)
print("RFR score=",score)
print("RFR F1=",f1)
svc = SVC()
svc.fit(XC,Y)
scores = cross_val_score(svc, XC, Y, scoring='accuracy',cv=10)
score = scores.mean()
predicted = cross_val_predict(svc, XC, Y,cv=10)
acc = accuracy_score(Y,predicted)
f1 = f1_score(Y,predicted)
print("SVC ACC=",acc)
print("SVC score=",score)
print("SVC F1=",f1)
knn = KNeighborsClassifier()
knn.fit(XC,Y)
scores = cross_val_score(knn, XC, Y, scoring='accuracy',cv=10)
score = scores.mean()
predicted = cross_val_predict(knn, XC, Y,cv=10)
acc = accuracy_score(Y,predicted)
f1 = f1_score(Y,predicted)
print("KNN ACC=",acc)
print("KNN score=",score)
print("KNN F1=",f1)

# Apartado 6
df = pd.read_excel("Churn_Modelling_NANs.xlsx", na_values='NA')
# Si hay que eliminar las filas con perdidos en una columna concreta
# Para los ejercicios opcionales
# Tipicamente se hace con la variable de salida
dfClean = df.dropna(subset='EXITED')
#Despues se haría la imputación 