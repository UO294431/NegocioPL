import matplotlib.pyplot as plt
from sklearn.svm import SVR
import numpy as np
import pandas as pd

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
X = X_full[:,:]
orden = np.argsort(Y)
horizontal = np.arange(Y.shape[0])

regressorL = SVR(kernel='linear',C=1e1,epsilon=1)
regressorL.fit(X, Y)
plt.scatter(horizontal, regressorL.predict(X)[orden], color='red', linewidth=3, label='Linear SVR')

regressorP = SVR(kernel='poly',C=1e1,epsilon=1)
regressorP.fit(X, Y)
plt.scatter(horizontal, regressorP.predict(X)[orden], color='blue', linewidth=3, label='Poly SVR')

regressorRBF = SVR(kernel='rbf',C=1e1,epsilon=1)
regressorRBF.fit(X, Y)
plt.scatter(horizontal, regressorRBF.predict(X)[orden], color='green', linewidth=3, label='RBF SVR')

plt.scatter(horizontal, Y[orden], color='black', label = 'Data')
plt.legend()
plt.show()

regressor0 = SVR(kernel='linear',C=1e1,epsilon=1)
regressor0.fit(X, Y)
plt.scatter(horizontal, regressor0.predict(X)[orden], color='red', linewidth=3, label='1')

regressor1 = SVR(kernel='linear',C=1e1,epsilon=1)
regressor1.fit(X, Y)
plt.scatter(horizontal, regressor1.predict(X)[orden], color='blue', linewidth=3, label='1e1')

regressor2 = SVR(kernel='rbf',C=1e1,epsilon=1)
regressor2.fit(X, Y)
plt.scatter(horizontal, regressor2.predict(X)[orden], color='green', linewidth=3, label='1e2')

regressor3 = SVR(kernel='rbf',C=1e1,epsilon=1)
regressor3.fit(X, Y)
plt.scatter(horizontal, regressor3.predict(X)[orden], color='yellow', linewidth=3, label='1e3')

plt.scatter(horizontal, Y[orden], color='black', label = 'Data')
plt.legend()
plt.show()