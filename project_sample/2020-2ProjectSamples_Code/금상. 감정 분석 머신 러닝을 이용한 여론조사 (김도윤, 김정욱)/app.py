import numpy as np
import chardet
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import time
import os

#DATASET
DATASET_ENCODING = "utf-8"
DATASET_COLUMNS = ["Username", "Comment"]
# KERAS
SEQUENCE_LENGTH = 300
# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

app = Flask(__name__)
model = keras.models.load_model('model.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE

def predict_sentiment(text, include_neutral=True):
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at}  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    positives = 0
    neutrals = 0
    negatives = 0
    storage = []
    inputs = [x for x in request.form.values()]
    os.system('python3 comment_scraper.py '+inputs[0])
    with open('results.csv', 'rb') as f:
        result = chardet.detect(f.read())  # or readline if the file is large
    df = pd.read_csv('results.csv', encoding=result['encoding'])
    df.head(5)
    
    for i in range(len(df['Comment'])):
        comment = df['Comment'][i]
        
        prediction = predict_sentiment(comment)
        if prediction['label'] == POSITIVE:
            positives += 1
        elif prediction['label'] == NEUTRAL:
            neutrals += 1
        elif prediction['label'] == NEGATIVE:
            negatives += 1
        else:
            print("DEBUG : UNEXPECTED LABEL")
        print(comment, prediction['label'])
        storage.append((comment, prediction['label'], prediction['score']))

    labels = 'Positive', 'Neutral', 'Negative'
    sizes = [positives, neutrals, negatives]
    colors = colors = ["#F7464A", "#46BFBD", "#FDB45C"]
    # fig1, ax1 = plt.subplots()
    # ax1.pie(sizes, labels=labels)
    # ax1.axis('equal')
    # plt.savefig('static/images/plot.png')

    pie_labels = labels
    pie_values = sizes

    max_sentiment = ''
    if positives >= neutrals and positives >= negatives:
        max_sentiment = 'Positive'
    elif neutrals >= positives and neutrals >= negatives:
        max_sentiment = 'Neutral'
    else:
        max_sentiment = 'Negative'

    return render_template('index.html', prediction_text='Overall sentiment of this video is {}!'.format(max_sentiment), url='/static/images/plot.png', set=zip(pie_values, pie_labels, colors))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
