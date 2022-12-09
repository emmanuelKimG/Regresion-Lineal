import pandas as pd  
import numpy as np  #para operaciones comp vectores
import matplotlib.pyplot as plt

datos = pd.read_csv("")
datos.head()

x= np.array(datos['x'])
y= np.array(datos['y'])

plt.plot(x,y,'oy')
plt.xlabel('x')
plt.ylabel('y')

m = np.size(x)

#Parametros iniciales 
a0 = 1
a1 = 0

iterMax = 100
beta = 0.02

h = a0+a1*x
plt.plot(x,h,'r') 
convergencia = np.zeros(iterMax)
J = (1/(2*m))*sum(np.power((h-y),2))
convergencia[0] = J
iter =0
while iter <iterMax:
    a0 =a0-beta*(1/m)*sum(h-y)
    a1= a1-beta*(1/m)*sum((h-y)*x)
    h=a0+a1*x
    J = (1/(2*m))*sum(np.power((h-y),2))
    convergencia[iter] = J
    iter +=1

plt.plot(x,h,'g')

plt.figure(2)
plt.plot(convergencia)

print("a0 =",a0,"a1= ",a1)

plt.show()



## Con paqueterias de python 




