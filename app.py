import flask
from flask import Flask, render_template, request,redirect,make_response
import nltk
import requests
import pickle 
import json



# with open(f'model/twitter_predictions.pkl', 'rb') as f:
#     model = pickle.load(f)

# with open(f'model/vectorizer.pkl', 'rb') as f:
#     vectorizer = pickle.load(f)
    
    
# app = flask.Flask(__name__, template_folder='templates')


# #Tokenizer
# from nltk.tokenize import RegexpTokenizer
# tokenizer = RegexpTokenizer(r'\w+')
    
    
# @app.route("/")
# def hello_world():
#      return render_template('index.html')
# @app.route('/predict', methods=['GET', 'POST'])
# def index():
    
#     if flask.request.method == 'GET':
#         return(flask.render_template('index.html'))
    
#     if flask.request.method == 'POST':
        
#         tweet = flask.request.form['tweet']


     
    
# if __name__ == "__main__":
#     app.run(debug=True)




#Package import
from flask import Flask, render_template, send_file, make_response, url_for, Response, redirect, request 
import tweepy
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import tweepy

import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.metrics import accuracy_score
from multiprocessing import Process

from transformers import pipeline
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer
 
#initialise app
app = Flask(__name__)

# decorator for homepage 
@app.route('/')
def home():
    return render_template('index.html',
                           PageTitle = "Landing page")
    
@app.route('/stock',methods =["GET", "POST"] )   
def index():
    if request.method == "POST":
       # getting input with name = fname in HTML form
       stock_symbol = request.form.get("s_symbol")
   
       result = twitter(stock_symbol)
    
    return render_template("index.html") 
@app.route('/login')
def log():
    return render_template('login.html')

def main_model(df):
    



#----------------------------------------------------------------------------------------------------------



    english_stopwords = stopwords.words('english')

    def remove_stopwords(tokens):
        return [word for word in tokens if word.lower() not in english_stopwords]

    stemmer = SnowballStemmer(language='english')

    def tokenize(text):
        return [stemmer.stem(word) for word in tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='tf')]

    vectorizer = CountVectorizer(lowercase=True, 
                                tokenizer=tokenizer,
                                stop_words=english_stopwords,
                                max_features=1000)
    #--------------------------------------------------------------------------------------------------------
    model = pkl.loads(open('model.pkl', 'rb'))

    #---------------------------------------------------------------------------------------------------------


    train_inputs = df['Tweet']
    vectorizer.fit(train_inputs)
    inputs = vectorizer.transform(train_inputs)
    preds = model.predict(inputs)
    dp = []
    dp2 = []
    #----------------------------------------------------------------------------------------------------------
    def graphs(dp,dp2):
        plt.rcParams['figure.figsize'] = [7.5, 3.5]
        plt.rcParams['figure.autolayout'] = True
        x = np.linspace(-2,2,10)
        plt.legend(['Predictions', 'Actual'])
        plt.plot(dp)
        plt.plot(dp2)
        savefig('graphs.png')
    
    def pie(dp,dp2,df5):
        df5 = pd.DataFrame({'labels': df5.index,'values': df5['sentiment']})
        labels = df5['labels']
        sizes = df5['values']
        colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','cyan','lightpink']
        patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90, pctdistance=1.1, labeldistance=1.2)
        plt.legend(patches, labels, loc="best")
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('chart.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.rcParams['figure.figsize'] = [7.5, 3.5]
        plt.rcParams['figure.autolayout'] = True
        x = np.linspace(-2,2,10)
        graphs(dp,dp2)
        # plt.plot(dp)
        # plt.plot(dp2)
        # plt.savefig('graphs.png', dpi=300, bbox_inches='tight')
    
    def iterate(inputs):
        inputs['Result'] = inputs['Tweet'].apply(lambda x: classifier(x))
        inputs['sentiment'] = inputs['Result'].apply(lambda x: (x[0]['label']))

        new_inputs = inputs['Tweet']
        new_targets = inputs['sentiment']

        vectorizer.fit(new_inputs)
        new_inputs = vectorizer.transform(new_inputs)

        new_preds = model.predict(new_inputs)
        print(pd.Series(new_preds).value_counts())
        print(pd.Series(new_targets).value_counts())

        a = pd.Series(new_preds).value_counts()
        b = pd.Series(new_targets).value_counts()

        dp.append(a[0])
        dp2.append(b[0])

        # graphs(dp,dp2)

        df5 = pd.DataFrame(pd.Series(new_targets).value_counts())

        print('Accuracy is : ',accuracy_score(new_targets, new_preds))
        pie(dp,dp2,df5)
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    groups = np.arange(len(df.iloc[:,3:4])) // 100
    for idx, subset in df.groupby(groups):
        iterate(pd.DataFrame(subset.Tweet))
        
    i=1
    while(i>0):
        p=Process()
        p.start()
        p.join()
        i+1  


def twitter(stock_symbol):
    api_key = "Zf5TkFjjbqxyzC02cFNbrfd2V"
    api_key_secret = "afUdvsB5rueFfZhAFguKlpbhzyFwL1rEp898MSj1HBBh7In8Sk"

    access_token = '1420374454392156165-XUOMDIw2YmAIcF1xZVyp4Y50JEGUlB'
    access_token_secret = 'rVOG17Kl7vZf8wB3cLt9X6YRk3LrEJctL48KyqX2SPqJI'

    ("Enter the stock code:")
    inp = stock_symbol
    # nos = input("Number of Tweets you want : ")
    nos=5000
    keyword = f'{inp}'
    inp1="stockdata"

    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    cursor = tweepy.Cursor(api.search_tweets, q=keyword, count=200,tweet_mode='extended').items(int(nos))
    data = []
    columns = ['Time', 'User', 'Tweet']
    for tweet in cursor:
        data.append([tweet.created_at, tweet.user.screen_name, tweet.full_text])

    df = pd.DataFrame(data, columns=columns)

    df.to_csv(f'{inp1}_tweets.csv') 
    dataframe = pd.DataFrame(f'{inp1}_tweets.csv')
    main_model(df)
    
   

if __name__ == '__main__':
    app.run(debug = True)
    
    