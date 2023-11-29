from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar el conjunto de datos
data = pd.read_csv("c:\\employee.csv")

# Verifica si hay datos faltantes y decide cómo tratarlos
data.isnull().sum()

# Aplicar codificación one-hot a las variables categóricas
data = pd.get_dummies(data, columns=["Education", "City", "Gender", "EverBenched"], drop_first=True)

# Supongamos que las etiquetas de clase son binarias (LeaveOrNot)
test_size = 0.2  # Proporción para el conjunto de prueba

# Dividir el conjunto de datos en entrenamiento y prueba manualmente
train_indices = []
test_indices = []

for class_label in [0, 1]:
    class_data = data[data["LeaveOrNot"] == class_label]
    num_samples = len(class_data)
    num_test_samples = int(test_size * num_samples)

    # Obtener los índices de prueba
    test_indices_class = np.random.choice(class_data.index, num_test_samples, replace=False)
    test_indices.extend(test_indices_class)

    # Los índices de entrenamiento son los que no están en los índices de prueba
    train_indices_class = [idx for idx in class_data.index if idx not in test_indices_class]
    train_indices.extend(train_indices_class)

# Crear conjuntos de entrenamiento y prueba
X_train_class = data.iloc[train_indices].drop("LeaveOrNot", axis=1)
y_train_class = data.iloc[train_indices]["LeaveOrNot"]
X_test_class = data.iloc[test_indices].drop("LeaveOrNot", axis=1)
y_test_class = data.iloc[test_indices]["LeaveOrNot"]

# Normalizar las características si es necesario (ya normalizadas)
mean = X_train_class.mean()
std = X_train_class.std()
X_train_class = (X_train_class - mean) / std
X_test_class = (X_test_class - mean) / std

# Agregar una columna de unos para el término independiente b0
X_train_class = np.column_stack((np.ones(len(X_train_class)), X_train_class))
X_test_class = np.column_stack((np.ones(len(X_test_class)), X_test_class))

# Inicializar los coeficientes (b0, b1, b2, etc.)
coefficients = np.zeros(X_train_class.shape[1])

# Definir la tasa de aprendizaje y el número de iteraciones
learning_rate = 0.01
num_iterations = 10000

# Función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Crear una lista para almacenar los valores de la función de costo en cada iteración
cost_history = []

# Entrenar el modelo de regresión logística
for i in range(num_iterations):
    z = np.dot(X_train_class, coefficients)
    predictions = sigmoid(z)
    gradient = np.dot(X_train_class.T, (predictions - y_train_class)) / len(y_train_class)
    coefficients -= learning_rate * gradient

    # Calcular la función de costo (log loss)
    cost = -np.mean(y_train_class * np.log(predictions) + (1 - y_train_class) * np.log(1 - predictions))
    cost_history.append(cost)

# Realizar predicciones en el conjunto de prueba
z = np.dot(X_test_class, coefficients)
predictions = sigmoid(z)
y_pred = (predictions >= 0.5).astype(int)

# Graficar la función de costo en función del número de iteraciones
plt.plot(range(num_iterations), cost_history)
plt.xlabel('Número de Iteraciones')
plt.ylabel('Función de Costo')
plt.title('Curva de Aprendizaje')
plt.show()

# Realizar predicciones en el conjunto de prueba y calcular la precisión del modelo
z = np.dot(X_test_class, coefficients)
predictions = sigmoid(z)
y_pred = (predictions >= 0.5).astype(int)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test_class, y_pred)
print("Precisión del modelo: {:.2f}%".format(accuracy * 100))

# Calcular otras métricas
precision = precision_score(y_test_class, y_pred)
recall = recall_score(y_test_class, y_pred)
f1 = f1_score(y_test_class, y_pred)

print("Precisión: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))

# Mostrar matriz de confusión
conf_matrix = confusion_matrix(y_test_class, y_pred)
print("Matriz de Confusión:")
print(conf_matrix)
