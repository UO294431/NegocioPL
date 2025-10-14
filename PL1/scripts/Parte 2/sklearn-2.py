import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import mean_squared_error
import pandas as pd

# boston_dataset = datasets.load_boston()
# print(boston_dataset.feature_names)
# X_full = boston_dataset.data
# Y = boston_dataset.target

# boston_dataset = datasets.load_boston()
# print(boston_dataset.feature_names)
# X_full = boston_dataset.data
# Y = boston_dataset.target


COLS_X = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
]
COL_Y = "MEDV"


def read_boston_statlib(path_or_url: str):
    """
    Lee el fichero 'boston' de StatLib (formato 2 líneas/registro) y devuelve:
    X : DataFrame (13 features)
    y : Series (MEDV)
    df: DataFrame completo (features + target)
    """
    # Lee todas las líneas de datos, omitiendo el encabezado largo del archivo
    raw = pd.read_csv(
        path_or_url, sep=r"\s+", header=None, skiprows=22, engine="python"
    )

    # Even rows (0,2,4,...) contienen 11 columnas: CRIM..PTRATIO
    even = raw.iloc[::2, :11].reset_index(drop=True)

    # Odd rows (1,3,5,...) contienen 3 columnas: B, LSTAT, MEDV
    odd = raw.iloc[1::2, :3].reset_index(drop=True)

    # Ensambla las 13 features y la etiqueta
    X = pd.concat([even, odd.iloc[:, :2]], axis=1)
    X.columns = COLS_X
    y = odd.iloc[:, 2].rename(COL_Y)

    df = X.copy()
    df[COL_Y] = y
    return X, y, df


data_url = "http://lib.stat.cmu.edu/datasets/boston"
X, y, raw_df = read_boston_statlib(data_url)
X_full = X.values
Y = y.values

print(raw_df.columns)

print(X_full.shape)
print(Y.shape)
# Se elige la variable mas dependiente de la salida
selector = SelectKBest(f_regression, k=1)
selector.fit(X_full, Y)
X = X_full[:, selector.get_support()]

regressor = LinearRegression()
regressor.fit(X, Y)
# Error cuadratico medio de cada fold, seguido de media de folds
scores = cross_val_score(regressor, X, Y, scoring='neg_mean_squared_error',cv=10)
score = scores.mean()
# TO DO: Usar un test estadistico para comparar los diferentes scores

# La prediccion es la respuesta del modelo aprendido en el
# el fold para el que la instancia fue parte del conjunto de test
predicted = cross_val_predict(regressor, X, Y,cv=10)
mse = mean_squared_error(Y,predicted)
# Resultados parecidos pero no iguales
print("LIN MSE=",mse)
print("LIN score=",-score)

regressor = SVR(kernel='rbf',C=1e1,epsilon=1)
regressor.fit(X, Y)
score = cross_val_score(regressor, X, Y, scoring='neg_mean_squared_error',cv=10).mean()
predicted = cross_val_predict(regressor, X, Y,cv=10)
mse = mean_squared_error(Y,predicted)
print("SVR MSE=",mse)
print("SVR score=",-score)

regressor = RandomForestRegressor()
regressor.fit(X, Y)
score = cross_val_score(regressor, X, Y, scoring='neg_mean_squared_error',cv=10).mean()
predicted = cross_val_predict(regressor, X, Y,cv=10)
mse = mean_squared_error(Y,predicted)
print("RNF MSE=",mse)
print("RNF score=",-score)
