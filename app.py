import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from autocorrect import Speller
from tensorflow.keras.models import load_model
import json
import random
from flask import Flask, render_template, request

lemmatizer = WordNetLemmatizer()
model = load_model('model/chatbot_model.h5')
spell = Speller()

intents = json.loads(open('Dataset\Dataset.json').read())
words = pickle.load(open('model\words.pkl', 'rb'))
classes = pickle.load(open('model\classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    if ints:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    
    return "Sorry, I didn't understand that. Could you please rephrase?"

def chatbot_response(msg):
    msg = spell(msg)

    res = getResponse(predict_class(msg, model), intents)
    print(f"chatbot_response: {res}")
    
    return res

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    if not userText:
         return "Please provide some input."
    
    chatbot_response_text = chatbot_response(userText)
    return chatbot_response_text

if __name__ == "__main__":
    app.run()
