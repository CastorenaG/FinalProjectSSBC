# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Leer el dataset zoo.csv
zoo_df = pd.read_csv("c:\\zoo.csv")

# Codificar las variables categóricas

zoo_df['hair'] = zoo_df['hair'].map({0: 0, 1: 1})

# Agregar una columna de unos para el sesgo (bias)
zoo_df["bias"] = 1

# Definir la función de distancia euclidiana
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# Definir la función k-NN para una sola instancia
def knn_predict(data, test_instance, k):
    distances = []
    for i in range(len(data)):
        x_data = data.iloc[i, :-1].values
        distance = euclidean_distance(test_instance, x_data)
        distances.append((distance, data.iloc[i, -1]))
    distances = sorted(distances, key=lambda x: x[0])[:k]
    neighbors = [neighbor[1] for neighbor in distances]
    return max(set(neighbors), key=neighbors.count)

# Dividir los datos en características (X) y etiquetas (y)
X = zoo_df.drop(['animal_name', 'class_type'], axis=1)  # Ajusta según tus columnas
y = zoo_df['class_type']

# Dividir los datos en conjuntos de entrenamiento y prueba
split_ratio = 0.8
split_index = int(len(zoo_df) * split_ratio)

X_train = X.iloc[:split_index, :]
y_train = y.iloc[:split_index]

X_test = X.iloc[split_index:, :]
y_test = y.iloc[split_index:]

# Definir el valor de k (número de vecinos)
k_value = 3

# Realizar predicciones en el conjunto de prueba
predictions = []
for i in range(len(X_test)):
    test_instance = X_test.iloc[i, :].values
    prediction = knn_predict(pd.concat([X_train, y_train], axis=1), test_instance, k_value)
    predictions.append(prediction)

# Evaluación del modelo (métricas de evaluación)
accuracy = accuracy_score(y_test, predictions) * 100  
precision = precision_score(y_test, predictions, average='weighted', zero_division=1)
recall = recall_score(y_test, predictions, average='weighted', zero_division=1)
f1 = f1_score(y_test, predictions, average='weighted', zero_division=1)
conf_matrix = confusion_matrix(y_test, predictions)

print("Precisión del modelo: {:.2f}%".format(accuracy))
print("Precisión (weighted):", precision)
print("Recall (weighted):", recall)
print("F1-score (weighted):", f1)
print("Matriz de Confusión:")
print(conf_matrix)
