import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("/content/drive/MyDrive/ProyectoFinalSemBasCon/zoo.csv")

# Separar las características (X) y la variable objetivo (y)
X = df.drop(['animal_name', 'class_type'], axis=1)
y = df['class_type']

# Convertir la variable objetivo a numpy array
y_array = y.values.reshape(-1, 1)

# Función de activación sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

# Inicialización de pesos y sesgos
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights_input_hidden = np.random.rand(input_size, hidden_size)
    weights_hidden_output = np.random.rand(hidden_size, output_size)
    bias_hidden = np.zeros((1, hidden_size))
    bias_output = np.zeros((1, output_size))
    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

# Dividir el conjunto de datos en entrenamiento y prueba
def split_data(data, test_size=0.2, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    indices = np.arange(len(data))
    np.random.shuffle(indices)
    test_set_size = int(len(data) * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]

# Entrenamiento del modelo
def train(X, y, learning_rate, epochs, hidden_size):
    input_size = X.shape[1]
    output_size = 1  # Para un problema de clasificación binaria

    weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = initialize_parameters(
        input_size, hidden_size, output_size
    )

    for epoch in range(epochs):
        # Feedforward
        hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        predicted_output = sigmoid(output_layer_input)

        # Calcular el error
        error = y - predicted_output

        # Backpropagation
        output_error = error * sigmoid_derivative(predicted_output)
        hidden_layer_error = output_error.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)

        # Actualizar pesos y sesgos
        weights_hidden_output += hidden_layer_output.T.dot(output_error) * learning_rate
        weights_input_hidden += X.T.dot(hidden_layer_error) * learning_rate
        bias_output += np.sum(output_error, axis=0, keepdims=True) * learning_rate
        bias_hidden += np.sum(hidden_layer_error, axis=0, keepdims=True) * learning_rate

    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

# Predicción
def predict(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Redondear las predicciones a 0 o 1 para un problema de clasificación binaria
    predictions = np.round(predicted_output)

    return predictions.flatten()

# Dividir el conjunto de datos manualmente
train_data, test_data = split_data(df, test_size=0.2, random_seed=42)

# Normalizar manualmente
X_train_normalized = (train_data.drop(['animal_name', 'class_type'], axis=1) - X.mean()) / X.std()
X_test_normalized = (test_data.drop(['animal_name', 'class_type'], axis=1) - X.mean()) / X.std()

# Agregar una columna de unos para el sesgo
X_train_normalized_bias = np.hstack((np.ones((X_train_normalized.shape[0], 1)), X_train_normalized))
X_test_normalized_bias = np.hstack((np.ones((X_test_normalized.shape[0], 1)), X_test_normalized))

# Parámetros de entrenamiento
learning_rate = 0.01
epochs = 1000
hidden_size = 5

# Entrenar el modelo
weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = train(
    X_train_normalized_bias, train_data['class_type'].values.reshape(-1, 1),
    learning_rate, epochs, hidden_size)

# Hacer predicciones en el conjunto de prueba
predictions = predict(X_test_normalized_bias, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)

# Calcular métricas
accuracy = accuracy_score(test_data['class_type'].values, predictions)
precision = precision_score(test_data['class_type'].values, predictions, average='micro')  # Puedes ajustar 'micro' según tus necesidades
recall = recall_score(test_data['class_type'].values, predictions, average='micro')  # Puedes ajustar 'micro' según tus necesidades
f1 = f1_score(test_data['class_type'].values, predictions, average='micro')  # Puedes ajustar 'micro' según tus necesidades
conf_matrix = confusion_matrix(test_data['class_type'].values, predictions)

# Mostrar las métricas
print("Accuracy en el conjunto de prueba:", accuracy)
print("Precisión en el conjunto de prueba:", precision)
print("Sensibilidad en el conjunto de prueba:", recall)
print("Puntuación F1 en el conjunto de prueba:", f1)
print("Matriz de confusión en el conjunto de prueba:\n", conf_matrix)
