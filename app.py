from flask import Flask, render_template, request
import numpy as np
import pickle
import spacy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

maxlen = 28
nlp = spacy.load('en_core_web_sm')



with open('bernoullinb_88.pkl', 'rb') as ml_model : 
    ml_model = pickle.load(ml_model)
with open('cv.pkl', 'rb') as ml_pre : 
    cv = pickle.load(ml_pre)



lstm_model = load_model('emotion_lstm.h5')
with open('tokenizer.pkl', 'rb') as lstm_pre : 
    tokenizer = pickle.load(lstm_pre)


def clean_text(text) : 
    return " ".join([token.lemma_ for token in nlp(text.lower()) if not (token.is_stop or token.is_punct or len(token) == 1 )])


def model_predict_lstm(text) : 
    # text = input("Enter text : ")
    test_corpus = [clean_text(text)]
    test_sequences = tokenizer.texts_to_sequences(test_corpus)
    test_with_pad_sequence = pad_sequences(test_sequences, maxlen=maxlen, padding='post', truncating='post')
    return test_with_pad_sequence



app = Flask(__name__)

emoji_map = {
    'joy' : '😂',
    'sadness' : '😩',
    'anger' : '😡',
    'fear' : '😨',
    'love' : '❤️',
    'surprise' : '😮', 
}

ml = {0: 'anger',1 : 'fear', 2 : 'joy', 3 : 'sadness'}
dl = {0 : 'anger', 1 : 'fear', 2 :'joy',3 :  'love', 4 : 'sadness', 5 :  'surprise'}

@app.route("/", methods = ['GET', "POST"] )
def Home() : 
    if request.method == "POST" : 
        ml_message = request.form['ml_message']
        
        if ml_message != "" : 
            sig_text = cv.transform([clean_text(ml_message)])
            sig_pred = ml_model.predict(sig_text)
            print(sig_pred)
            return render_template('index.html', data = ml_message, model_type = "According To Naive Bayes Model", pred = [(sig_pred[0], 100)], emotion_type = ml, emoji_map = emoji_map)
        else : 
            return render_template('index.html')

    return render_template('index.html')


@app.route("/deep-learning", methods = ['GET', "POST"] )
def deepLearning() : 
    if request.method == "POST" : 
        dl_message = request.form['dl_message']
        
        if dl_message != "" : 
            sig_num = model_predict_lstm(dl_message)
            sig_pred = lstm_model.predict(sig_num)
            pred_arr = sorted(enumerate(sig_pred[0]), key = lambda x : x[1], reverse = True)
            pred_arr = [(itm[0], round(itm[1]*100, 2)) for itm in pred_arr]
            print(pred_arr)
            return render_template('index.html', data = dl_message, model_type = "According To LSTM Model", pred = pred_arr, emotion_type = dl, emoji_map = emoji_map)
        else : 
            render_template('index.html')

    return render_template('index.html')


if __name__ == "__main__" : 
    app.run(debug=True, host='0.0.0.0')