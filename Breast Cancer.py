from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

ruta = r'C:\Users\rockn\OneDrive\Desktop\breast_cancer.csv'


df.to_csv(ruta, index=False)
print('Archivo guardado en: ', ruta)