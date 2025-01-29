# Importar librerías necesarias
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Cargar el dataset de correos electrónicos
newsgroups = fetch_20newsgroups(subset='train', categories=['sci.space', 'rec.sport.baseball'])
print(newsgroups)
X, y = newsgroups.data, newsgroups.target

# Vectorizar el texto
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo
modelo = MultinomialNB()
modelo.fit(X_train, y_train)

# Hacer predicciones
y_pred = modelo.predict(X_test)

# Evaluar el modelo
print("Accuracy del modelo:", accuracy_score(y_test, y_pred))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
