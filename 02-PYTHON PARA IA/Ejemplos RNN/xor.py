# Esta red neuronal binaria hace una predicion de una compuerta XOR con base a sus input en matriz de Dataset
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# cargamos las 4 combinaciones de las compuertas XOR en el dataset de entrenamiento
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

# y estos son los resultados que se obtienen, en el mismo orden en el dataset de prueba o resultado
target_data = np.array([[0],[1],[1],[0]], "float32")

# Se crea el modelo con 3 capaz, 2 capaz ocultas , 16 en la oculta y una de salida
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Se compila el modelo con una metrica binaria y perdida de error media cuadrado
model.compile(loss='mean_squared_error',
optimizer='adam',
metrics=['binary_accuracy'])

# Se entrena el modelo con mil iteraciones de aprendizaje
model.fit(training_data, target_data, epochs=650)

# evaluamos el modelo
scores = model.evaluate(training_data, target_data)

# Se imprime la metrica usada , el porcentaje exito y se hace la prediccion con base a la matriz XOR
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print (model.predict(training_data).round())