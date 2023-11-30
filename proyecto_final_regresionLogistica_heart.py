import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix

# Cargar el conjunto de datos
data = pd.read_csv("c:\\heart.csv")

X = data.drop("output", axis=1)
y = data["output"]

# Obtener los índices para el conjunto de prueba
test_size = 0.2
test_indices = []
for class_label in [0, 1]:
    class_data = data[data["output"] == class_label]
    num_samples = len(class_data)
    num_test_samples = int(test_size * num_samples)

    # Obtener los índices de prueba
    test_indices_class = np.random.choice(class_data.index, num_test_samples, replace=False)
    test_indices.extend(test_indices_class)

# Los índices de entrenamiento son los que no están en los índices de prueba
train_indices = [idx for idx in data.index if idx not in test_indices]

# Crear conjuntos de entrenamiento y prueba
X_train = X.loc[train_indices]
y_train = y.loc[train_indices]
X_test = X.loc[test_indices]
y_test = y.loc[test_indices]

# Normalizar las características
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_train.mean()) / X_train.std()

# Inicializar los coeficientes (b0, b1, b2, etc.)
coefficients = np.zeros(X_train.shape[1] + 1)

# Agregar una columna de unos para el término independiente b0
X_train = np.column_stack((np.ones(len(X_train)), X_train))
X_test = np.column_stack((np.ones(len(X_test)), X_test))

# Definir la tasa de aprendizaje y el número de iteraciones
learning_rate = 0.001  
num_iterations = 10000  

# Función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Crear una lista para almacenar los valores de la función de costo en cada iteración
cost_history = []

# Entrenar el modelo de regresión logística
for i in range(num_iterations):
    z = np.dot(X_train, coefficients)
    predictions = sigmoid(z)
    gradient = np.dot(X_train.T, (predictions - y_train)) / len(y_train)
    coefficients -= learning_rate * gradient

    # Calcular la función de costo (log loss)
    cost = -np.mean(y_train * np.log(predictions) + (1 - y_train) * np.log(1 - predictions))
    cost_history.append(cost)

# Realizar predicciones en el conjunto de prueba
z = np.dot(X_test, coefficients)
predictions = sigmoid(z)
y_pred = (predictions >= 0.5).astype(int)

# Graficar la función de costo en función del número de iteraciones
plt.plot(range(num_iterations), cost_history)
plt.xlabel('Número de Iteraciones')
plt.ylabel('Función de Costo')
plt.title('Curva de Aprendizaje')
plt.show()

# Calcular métricas
accuracy = np.mean(y_test == y_pred)
precision = np.sum((y_test == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
recall = np.sum((y_test == 1) & (y_pred == 1)) / np.sum(y_test == 1)
f1 = 2 * (precision * recall) / (precision + recall)

print("Precisión del modelo: {:.2f}%".format(accuracy * 100))
print("Precisión: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))

# Calcular y mostrar la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(conf_matrix)
