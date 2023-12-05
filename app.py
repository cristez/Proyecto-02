from flask import Flask, request, render_template
import joblib
import spacy
import re

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

app = Flask(__name__)

# Cargar el modelo
modelo = joblib.load('modelo_random_forest.pkl')
vectorizer = joblib.load('vectorizer.pkl')
nlp = spacy.load('es_core_news_sm')

@app.route('/', methods=['GET', 'POST'])
def index():
    calificacion = None
    if request.method == 'POST':
        nuevo_ensayo = request.form['ensayo']
        ensayo = nuevo_ensayo
                # Ejemplo de limpieza de texto (debería ser igual a como limpiaste los datos de entrenamiento)
        
        if len(nuevo_ensayo.split()) > 0 and len(nuevo_ensayo.split()) < 70:
    # Manejar el caso de un ensayo demasiado corto, por ejemplo, devolviendo un mensaje de error
             return render_template('index.html', calificacion="El ensayo es demasiado corto!", ensayo=ensayo)
        elif len(nuevo_ensayo.split()) == 0:
             return render_template('index.html', calificacion="¡Ingrese su ensayo!", ensayo=ensayo)

        
        nuevo_ensayo = nuevo_ensayo.lower()
        nuevo_ensayo = re.sub(r'\W', ' ', nuevo_ensayo)
        nuevo_ensayo = re.sub(r'\s+', ' ', nuevo_ensayo)


        # Aplicar lematización 
        def lemmatize_text(text):
            doc = nlp(text)
            return ' '.join([token.lemma_ for token in doc])

        nuevo_ensayo = lemmatize_text(nuevo_ensayo)

# Transformar con el vectorizador

        ensayo_vectorizado = vectorizer.transform([nuevo_ensayo])
        calificacion = modelo.predict(ensayo_vectorizado)
        calificacion = round(calificacion[0], 1)  # Redondear la calificación si es necesario

    return render_template('index.html', calificacion=calificacion, ensayo=ensayo)

if __name__ == '__main__':
    app.run(debug=True)
