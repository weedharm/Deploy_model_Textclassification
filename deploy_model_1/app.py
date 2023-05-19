from flask import Flask, request, render_template

app = Flask(__name__)


import tensorflow as tf
model = tf.keras.models.load_model('my_model.h5')
import json
data_train = 'train_intents.tsv'
import pandas as pd
data_train = pd.read_table(data_train, header=None)
text_train = data_train[0].values
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf.fit(text_train)
X_train_tf_idf = tfidf.transform(text_train)
from sklearn.decomposition import TruncatedSVD
svd_tf_idf= TruncatedSVD(n_components=100, random_state=42)
svd_tf_idf.fit(X_train_tf_idf)
X_train_tf_idf_svd = svd_tf_idf.transform(X_train_tf_idf)
with open('intent2index.json') as f:
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
    app.run()