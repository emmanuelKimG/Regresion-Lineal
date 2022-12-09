import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import linear_model

datos = pd.read_csv("dataset_RegresionLineal.csv")

x=np.array(datos['x'])
x= x.reshape(-1,1)
y=np.array(datos['y'])
plt.plot(x,y,'oy')

regresion = linear_model.LinearRegression()
regresion.fit(x,y)     #encuentra los datos a0 y a1
h = regresion.predict(x)   #calcula la h final

plt.plot(x,h,'g')
print("a0 :",regresion.intercept_, "a1 :",regresion.coef_[0])   #a0 intecept  a1 coef

