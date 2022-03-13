from flask import Flask,render_template,url_for,request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


app=Flask(__name__)
Swagger(app)


final_model = pickle.load(open('models/log_model_final.pkl','rb'))
word_vector = pickle.load(open('models/bagg_word_final.pkl','rb'))




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/memeber')
def members():
    return render_template('members.html')


@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        Reviews = request.form['Reviews']

        data = [Reviews]
        print(data)
        vect = word_vector.transform(data)
        my_prediction = final_model.predict(vect)[0]

    return render_template('output.html',prediction = my_prediction)



if __name__ == '__main__':
    app.run(debug=True)
