import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ Definir los datos de entrada
X = np.array([[2, 3], [1, 1], [4, 2], [3, 6], [5, 5], [6, 2], [7, 3], [8, 6]])
y = np.array([1, 0, 1, 1, 0, 0, 1, 1])

# Agregar el término de sesgo
X_bias = np.c_[np.ones(X.shape[0]), X]  # Añadir columna de unos

# 2️⃣ Implementar el Perceptrón
def heaviside(z):
    return 1 if z >= 0 else 0

def predict(X, W):
    return np.array([heaviside(np.dot(W, x)) for x in X])

def train_perceptron(X, y, lr=0.1, epochs=100):
    W = np.random.randn(X.shape[1])  # Inicialización aleatoria de pesos
    
    for epoch in range(epochs):
        errors = 0
        for i in range(len(y)):
            y_pred = predict([X[i]], W)[0]
            error = y[i] - y_pred
            if error != 0:
                W += lr * error * X[i]
                errors += 1
        
        if errors == 0:  # Si no hay errores, terminamos antes
            break
    
    return W

# Entrenar el modelo
W = train_perceptron(X_bias, y)
print(W)

# 3️⃣ Visualizar los Resultados
plt.figure(figsize=(8, 6))

# Graficar puntos
for i, label in enumerate(y):
    if label == 0:
        plt.scatter(X[i, 0], X[i, 1], color='red', marker='o', label='Clase 0' if i == 0 else "")
    else:
        plt.scatter(X[i, 0], X[i, 1], color='blue', marker='s', label='Clase 1' if i == 0 else "")

# Dibujar la línea de decisión
x1_range = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
x2_range = -(W[0] + W[1] * x1_range) / W[2]  # Ecuación de la frontera de decisión
plt.plot(x1_range, x2_range, 'k-', label="Frontera de decisión")

# Etiquetas y leyenda
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Perceptrón: Clasificación de Puntos en 2D")
plt.legend()
plt.grid()
plt.show()
