import numpy as np
from flask import Flask, request,render_template
import os
import pickle
import flask
import os
import newspaper
from newspaper import Article
import urllib

import nltk
nltk.download('punkt')
app = Flask(__name__)
app=flask.Flask(__name__,template_folder='templates')
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def main():
    return render_template('main.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    url =request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    news = article.summary

    pred = model.predict([news])
    return render_template('main.html', prediction_text='The give news is "{}"'.format(pred[0]))    
if __name__=="__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)