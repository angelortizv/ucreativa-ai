# Importar librerías necesarias
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score

# Cargar el dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Filtrar solo imágenes de gatos (3) y perros (5)
train_filter = np.isin(y_train.flatten(), [3, 5])
test_filter = np.isin(y_test.flatten(), [3, 5])

x_train, y_train = x_train[train_filter], y_train[train_filter]
x_test, y_test = x_test[test_filter], y_test[test_filter]

# Convertir las etiquetas a índices de 0 y 1 para gatos y perros
y_train = np.where(y_train == 3, 0, 1)
y_test = np.where(y_test == 3, 0, 1)

# Preprocesar los datos
y_train = to_categorical(y_train, 2)  # 2 categorías: gato (0) y perro (1)
y_test = to_categorical(y_test, 2)    # 2 categorías: gato (0) y perro (1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Crear el modelo
modelo = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

# Compilar el modelo
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
modelo.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Evaluar el modelo
y_pred = modelo.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("Precisión del modelo:", accuracy_score(y_true_classes, y_pred_classes))
print("Reporte de clasificación:\n", classification_report(y_true_classes, y_pred_classes))
