# Este ejemplo es para usar un dataset de imagenes y crear una RED NEURONAL que identifique una imagen de un digito entre 1 y 10
# importa queras y el dataset mnist para cargar datos de entrenamiento y prueba
import keras
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data ()

# imprime la imagen 9 del dataset
import matplotlib.pyplot as plt
plt.imshow(x_train[9], cmap=plt.cm.binary)

# Numero de dimensiones tensor entrenamineto
print(x_train.ndim)

# Datos por cada eje del tensor
print(x_train.shape)

# Tipo de dato del tensor , el tipo de dato uint es de tamaño maximo 255
print(x_train.dtype)

# convierte los datos de entrenamiento y prueba a decimal
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

# Convierte a vector las imagenes 2d para usarlo como capa de entrada
x_train = x_train.reshape(60000, 784)

x_test = x_test.reshape(10000, 784)

# Aqui se imprime los tamaños en formato matriz de 2 dimensiones, con la cantidad total de pixeles de cada imagen
print(x_train.shape)
print(x_test.shape)

# Muestra valores del conjunto de datos y de entrenamiento y prueba
print(y_test[0])
print(y_train[0])
print(y_train.shape)
print(x_test.shape)

# importa to_categorical para transformar el numero de imagen a vector de tamaño 10 (Basicamente en una dimension)
from tensorflow.keras.utils import to_categorical

# Aqui transforma al vector binario de tamaño 10
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# muestra convertido a vector el 7 y 5 de las celdas anteriores
print(y_test[0])
print(y_train[0])

# Muestra el nuevo tamaño con base al vector de tamaño 10
print(y_train.shape)
print(y_test.shape)

# Aqui se crea el modelo de Red neuronal secuencial densa

from keras.models import Sequential
model = Sequential()

# Importa y crea el modelo secuencial de capaz
from keras.layers import Dense, Activation
# aqui se crear las 2 capas de la red con el metodo add() y la funcion sigmoid de aprendiza con capas de 784 y 10 nodos respectivamente
model.add(Dense(10, activation="sigmoid", input_shape=(784,)))
model.add(Dense(10, activation="softmax"))

# El metodo sumary() indica la arquitectura de la red neuronal, de la primera capa 785*10 y 10*10+10 de la segunda mas los 10 sesgos
model.summary()

# Se define la compilacion del modelo con la funcion de estimacion de perdida, el optimizador Descenso de gradiente y una metrica de precisio
model.compile(loss="categorical_crossentropy",
optimizer="sgd",
metrics = ["accuracy"])

# OJO aqui se entrena el modelo de red neuronal, con los dataset, se le indica usar 100 datos por actualizacion y que ejecute 5 iteraciones
model.fit(x_train, y_train, batch_size=300, epochs=85)

# Evaluacion del modelo con los datos de x_test y y_test
test_loss, test_acc = model.evaluate(x_test, y_test)

# Imprime la presicion de la prueba de la red neuronal
print("Test accuracy:", test_acc)

# La prediccion genera un vector o matriz de valore sobre x_test
predictions = model.predict(x_test)
print(predictions)

# argmax da la mayor probabilidad de pertenencia del numero en el vector
import numpy as np
np.argmax(predictions[11])

# Devuelve el vector de las probabilidades de pertenencia del numero
print(predictions[11])

# Comprueba que la suma total de probabilidades del vector es 1
np.sum(predictions[11])