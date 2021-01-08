import numpy as np
from sklearn.linear_model import LinearRegression

# Declarando X y Y como arrays
x = np.array([5,10,25,35,40,55]).reshape((-1,1))
y = np.array([5,15,20,32,42,52])
print(x)
print(y)

# Aplicando al variable MODEL para usar la regresion lineal
model = LinearRegression()
model.fit(x,y)

# Calculando el coeficiente, el interceptor y el slope
r_sq = model.score(x,y)
print("Coefficient of determination:", r_sq)
print("Intercepter:",model.intercept_)
print("Slope:",model.coef_)

# Una prediccion de x sobre y
y_pred = model.predict(x)
print("Prediction:", y_pred, sep="\n")