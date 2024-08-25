import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from autocorrect import Speller
from tensorflow.keras.models import load_model
import json
import random
from flask import Flask, render_template, request, redirect, url_for

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

with open("tests.json") as file:
    tests = json.load(file)

def get_questions(title):
    for test in tests["tests"]:
        if test["title"] == title:
            return test["questions"]
    return "Test not found"

def get_test_messages(title, score):
    score = int(score)
    message = ""
    if title.lower() == "depression test":  # depression test
        if score > 20:
            message = "Depression Test: Severe Depression"
        elif score > 15:
            message = "Depression Test: Moderately Severe Depression"
        elif score > 10:
            message = "Depression Test: Moderate Depression"
        elif score > 5:
            message = "Depression Test: Mild Depression"
        else:
            message = "Depression Test: No Depression"
        message += (
            " - Score: "
            + str(score)
            + "/27 (Your responses indicate that you may be at risk of harming yourself. If you need immediate help, you can reach the mental health service by visiting this link: https://www.thelivelovelaughfoundation.org/find-help/helplines)"
        )
    elif title.lower() == "anxiety test":  # anxiety test
        if score > 15:
            message = "Anxiety Test: Severe Anxiety"
        elif score > 10:
            message = "Anxiety Test: Moderate Anxiety"
        elif score > 5:
            message = "Anxiety Test: Mild Anxiety"
        else:
            message = "Anxiety Test: No Anxiety"
        message += " - Score: " + str(score) + "/21"
    else:
        message = "Test Title not found"
    message += ". These results are not meant to be a diagnosis. You can meet with a doctor or therapist to get a diagnosis and/or access therapy or medications. Sharing these results with someone you trust can be a great place to start!"
    return message

@app.route('/test')
def index():
    return render_template('test-home.html')

@app.route('/test/<title>', methods=['GET', 'POST'])
def test(title):
    if request.method == 'POST':
        score = request.form.get('score')
        message = get_test_messages(title, score)
        return render_template('result.html', title=title, score=score, message=message)
    
    questions = get_questions(title)
    if questions == "Test not found":
        return redirect(url_for('test-home'))
    
    return render_template('test.html', title=title, questions=questions)

if __name__ == "__main__":
    app.run()
