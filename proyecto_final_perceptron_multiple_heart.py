import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Cargar el conjunto de datos del corazón
df_heart = pd.read_csv("c:\\heart.csv")

np.random.seed(42)  
indices = np.arange(len(df_heart))
np.random.shuffle(indices)

# Proporción para el conjunto de prueba
test_size = 0.2
num_test_samples = int(test_size * len(df_heart))

# Obtener los índices de prueba
test_indices = indices[:num_test_samples]
train_indices = indices[num_test_samples:]

# Crear conjuntos de entrenamiento y prueba
train_data_heart = df_heart.iloc[train_indices]
test_data_heart = df_heart.iloc[test_indices]

# Separar las características (X) y la variable objetivo (y) para entrenamiento y prueba
X_train_heart = train_data_heart.drop('output', axis=1)
y_train_heart = train_data_heart['output']

X_test_heart = test_data_heart.drop('output', axis=1)
y_test_heart = test_data_heart['output']

# Normalizar 
mean_heart = X_train_heart.mean()
std_heart = X_train_heart.std()

X_train_normalized_heart = (X_train_heart - mean_heart) / std_heart
X_test_normalized_heart = (X_test_heart - mean_heart) / std_heart

# Agregar una columna de unos para el sesgo
X_train_normalized_bias_heart = np.hstack((np.ones((X_train_normalized_heart.shape[0], 1)), X_train_normalized_heart))
X_test_normalized_bias_heart = np.hstack((np.ones((X_test_normalized_heart.shape[0], 1)), X_test_normalized_heart))

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

# Función de costo (log loss)
def compute_cost(y, predicted_output):
    return -np.mean(y * np.log(predicted_output) + (1 - y) * np.log(1 - predicted_output))

# Predicción
def predict(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Redondear las predicciones a 0 o 1 para un problema de clasificación binaria
    predictions = np.round(predicted_output)

    return predictions.flatten()

# Entrenamiento del modelo
def train(X, y, learning_rate, epochs, hidden_size):
    input_size = X.shape[1]
    output_size = 1  # Para un problema de clasificación binaria

    weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = initialize_parameters(
        input_size, hidden_size, output_size
    )

    cost_history = []

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

        # Calcular la función de costo y almacenar en la lista de historial
        cost = compute_cost(y, predicted_output)
        cost_history.append(cost)

    # Graficar la función de costo a lo largo de las iteraciones
    plt.plot(range(epochs), cost_history)
    plt.xlabel('Número de Iteraciones')
    plt.ylabel('Función de Costo')
    plt.title('Curva de Aprendizaje')
    plt.show()

    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

# Parámetros de entrenamiento
learning_rate_heart = 0.01
epochs_heart = 1000
hidden_size_heart = 5

# Entrenar el modelo
weights_input_hidden_heart, weights_hidden_output_heart, bias_hidden_heart, bias_output_heart = train(
    X_train_normalized_bias_heart, y_train_heart.values.reshape(-1, 1),
    learning_rate_heart, epochs_heart, hidden_size_heart)

# Hacer predicciones en el conjunto de prueba
predictions_heart = predict(X_test_normalized_bias_heart, weights_input_hidden_heart, weights_hidden_output_heart, bias_hidden_heart, bias_output_heart)

# Métricas de evaluación
accuracy_heart = accuracy_score(y_test_heart, predictions_heart)
print("Precisión del modelo: {:.2f}%".format(accuracy_heart * 100))

precision_heart = precision_score(y_test_heart, predictions_heart)
print("Precisión: {:.2f}".format(precision_heart))

recall_heart = recall_score(y_test_heart, predictions_heart)
print("Recall: {:.2f}".format(recall_heart))

f1_heart = f1_score(y_test_heart, predictions_heart)
print("F1 Score: {:.2f}".format(f1_heart))

conf_matrix_heart = confusion_matrix(y_test_heart, predictions_heart)
print("Matriz de Confusión:")
print(conf_matrix_heart)
