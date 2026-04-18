import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# cargar datos
df = pd.read_csv('breast_cancer.csv')

X = df.drop('target', axis=1)
y = df['target']

# escalar (muy importante antes de PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# reducir a 2 dimensiones
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# separar clases
clase_0 = X_pca[y == 0]
clase_1 = X_pca[y == 1]

# gráfica
plt.figure()

plt.scatter(clase_0[:,0], clase_0[:,1], color='blue', s=40, alpha=0.5)
plt.scatter(clase_1[:,0], clase_1[:,1], color='red', s=40, alpha=0.5)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Dispersión de datos (PCA)')
plt.legend()
plt.grid()
print("Varianza explicada:", pca.explained_variance_ratio_)
plt.show()
'''
Los componentes principales corresponden a nuevas variables generadas mediante PCA que representan combinaciones lineales
de las variables originales.

El primer componente captura la mayor variabilidad de los datos
El segundo captura la segunda mayor variabilidad, siendo ortogonal al primero

Esto permite visualizar los datos de alta dimensionalidad en un espacio bidimensional

Al analizar la gráfica, se observa que se presenta un cierto grado de separación entre los diagnósticos, 
aunque existen zonas de traslape.
Pequeños valores de k pueden ser sensibles al ruido, mientras que valores intermedios podrían generalizar mejor al modelo.

Para la evaluación, se utilizan 3 rangos de casos para tener un rango de posibles valores de k

Caso 1: Las clases estan bien separadas, lo que quiere decir que k estará dentro de 1 y 3
Caso 2: Las clases se encuentran mezcladas, k tendrá valores entre 5 y 9
Caso 3: Las clases están muy mezcladas, no lo que genera mucho ruido. k tendrá valores mayores a 9
'''