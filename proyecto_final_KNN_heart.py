import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import shuffle

# Leer el dataset heart.csv
heart_df = pd.read_csv("c:\\heart.csv")

# Manualmente seleccionar índices para el conjunto de entrenamiento y prueba
np.random.seed(42)  
indices = np.arange(len(heart_df))
np.random.shuffle(indices)

# Proporción para el conjunto de prueba
test_size = 0.2
num_test_samples = int(test_size * len(heart_df))

# Obtener los índices de prueba
test_indices = indices[:num_test_samples]
train_indices = indices[num_test_samples:]

# Crear conjuntos de entrenamiento y prueba
train_data_heart = heart_df.iloc[train_indices]
test_data_heart = heart_df.iloc[test_indices]

# Separar las características (X) y la variable objetivo (y) para entrenamiento y prueba
X_train_heart = train_data_heart.drop('output', axis=1)
y_train_heart = train_data_heart['output']

X_test_heart = test_data_heart.drop('output', axis=1)
y_test_heart = test_data_heart['output']

# Normalizar manualmente
mean_heart = X_train_heart.mean()
std_heart = X_train_heart.std()

X_train_normalized_heart = (X_train_heart - mean_heart) / std_heart
X_test_normalized_heart = (X_test_heart - mean_heart) / std_heart

# Agregar una columna de unos para el sesgo
X_train_normalized_bias_heart = np.hstack((np.ones((X_train_normalized_heart.shape[0], 1)), X_train_normalized_heart))
X_test_normalized_bias_heart = np.hstack((np.ones((X_test_normalized_heart.shape[0], 1)), X_test_normalized_heart))

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
X = heart_df.drop(['output'], axis=1)
y = heart_df['output']

# Mezclar los datos para manejar el desequilibrio de clases
X, y = shuffle(X, y, random_state=42)

# Dividir los datos en conjuntos de entrenamiento y prueba
split_ratio = 0.8
split_index = int(len(heart_df) * split_ratio)

X_train = X.iloc[:split_index, :]
y_train = y.iloc[:split_index]

X_test = X.iloc[split_index:, :]
y_test = y.iloc[split_index:]

# Definir el valor de k (número de vecinos)
k_value = 5  # Experimenta con diferentes valores de k

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

# Visualización de la matriz de confusión con Matplotlib
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.colorbar()

classes = ['Clase 0', 'Clase 1']  # Ajusta según las clases de tu problema
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

plt.xlabel('Predicciones')
plt.ylabel('Valores reales')

# Añadir anotaciones en cada celda
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment='center', verticalalignment='center')

plt.show()  
