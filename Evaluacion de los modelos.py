import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

df = pd.read_csv('breast_cancer.csv')

def kNN(df):
    
    X = df.drop('target', axis=1)
    y = df['target']

    valores_k = [1, 3, 5, 7, 9, 11]  # lista, no set
    resultados = []

    for k in valores_k:
        modelo = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=k, metric='euclidean'))
        ])
        
        accuracy = cross_val_score(modelo, X, y, cv=5, scoring='accuracy').mean()
        precision = cross_val_score(modelo, X, y, cv=5, scoring='precision').mean()
        recall = cross_val_score(modelo, X, y, cv=5, scoring='recall').mean()
        
        resultados.append((k, accuracy, precision, recall))

    print('Resultados:')
    for r in resultados:
        print(f'k={r[0]} | Accuracy={r[1]:.4f} | Precision={r[2]:.4f} | Recall={r[3]:.4f}')

    mejor = max(resultados, key=lambda x: x[1])
    print('\nEl mejor modelo es:')
    print(f'k = {mejor[0]} con Accuracy = {mejor[1]:.4f}')

    ks = [r[0] for r in resultados]
    acc = [r[1] for r in resultados]
    prec = [r[2] for r in resultados]
    rec = [r[3] for r in resultados]

    plt.figure()
    plt.plot(ks, acc, marker='o', label='Accuracy')
    plt.plot(ks, prec, marker='o', label='Precision')
    plt.plot(ks, rec, marker='o', label='Recall')

    plt.xlabel('Valor de K')
    plt.ylabel('Desempeño')
    plt.title('Evaluación de k en kNN')
    plt.legend()
    plt.grid()

    plt.show()
    resultados_df = pd.DataFrame(resultados, columns=['k', 'accuracy', 'precision', 'recall'])
    resultados_df.to_csv('resultados_knn.csv', index=False)
kNN(df)