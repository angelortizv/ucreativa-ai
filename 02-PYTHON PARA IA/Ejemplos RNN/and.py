import numpy as np

# Datos de entrada (X) y sus etiquetas (y) para la compuerta lógica AND
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # Salida esperada (0 AND 0 = 0, etc.)

# Inicializamos pesos y bias aleatoriamente
np.random.seed(42)
weights = np.random.rand(2)  # Dos pesos (para dos entradas)
bias = np.random.rand(1)[0]  # Un bias
learning_rate = 0.1
epochs = 10

# Función de activación (escalón)
def step_function(z):
    return 1 if z >= 0 else 0

# Entrenamiento del perceptrón
for epoch in range(epochs):
    for i in range(len(X)):
        z = np.dot(X[i], weights) + bias  # Producto punto + bias
        y_pred = step_function(z)  # Aplicamos función escalón
        
        # Actualizamos los pesos y bias si hay error
        error = y[i] - y_pred
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

# Prueba del perceptrón entrenado
for i in range(len(X)):
    z = np.dot(X[i], weights) + bias
    y_pred = step_function(z)
    print(f"Entrada: {X[i]} -> Salida predicha: {y_pred}")
