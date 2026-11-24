from flask import Flask, request, render_template

app = Flask(__name__)

import os
# Resolve data files next to this script so the app works from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import tensorflow as tf
model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'my_model.h5'))
import json
# TF-IDF + TruncatedSVD fitted once on the original training text (the features the
# model was trained on) and saved, so predictions do not depend on re-fitting at startup.
# Personal names in the vocabulary were replaced by fictional ones; see README.
import joblib
_pre = joblib.load(os.path.join(BASE_DIR, 'preprocess.joblib'))
tfidf, svd_tf_idf = _pre['tfidf'], _pre['svd']
with open(os.path.join(BASE_DIR, 'intent2index.json')) as f:
    data = json.load(f) 


# returns keys of dictionary when you know the value
def get_key(val):
    for key, value in data.items():
         if val == value:
             return key


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/double', methods=['POST'])
def double():
    data = request.get_json()
    number = str(data['number'])
    result = get_key(model.predict(svd_tf_idf.transform(tfidf.transform([number]))).argmax(axis=-1))
    return f'{result}'

if __name__ == '__main__':
    app.run(host=os.environ.get('HOST', '127.0.0.1'), port=int(os.environ.get('PORT', '5000')))