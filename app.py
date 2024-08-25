import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from autocorrect import Speller
from tensorflow.keras.models import load_model
import json
import random
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from flask import Flask, render_template, request

nltk.download('popular')
lemmatizer = WordNetLemmatizer()
model = load_model('model\chatbot_model.h5')
spell = Speller()
nlp = spacy.load("en_core_web_sm")

eng_hi_model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
eng_hi_tokenizer = AutoTokenizer.from_pretrained(eng_hi_model_checkpoint)
eng_hi_model = AutoModelForSeq2SeqLM.from_pretrained(eng_hi_model_checkpoint)

eng_hi_translator = pipeline(
    "text2text-generation",
    model=eng_hi_model,
    tokenizer=eng_hi_tokenizer,
)

hi_eng_model_checkpoint = "Helsinki-NLP/opus-mt-hi-en"
hi_eng_tokenizer = AutoTokenizer.from_pretrained(hi_eng_model_checkpoint)
hi_eng_model = AutoModelForSeq2SeqLM.from_pretrained(hi_eng_model_checkpoint)

hi_eng_translator = pipeline(
    "text2text-generation",
    model=hi_eng_model,
    tokenizer=hi_eng_tokenizer,
)

def get_lang_detector(nlp, name):
    return LanguageDetector()

nlp = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe("language_detector", last=True)

def hinglish_to_hindi(text):
    hindi_text = transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
    return hindi_text

intents = json.loads(open('Dataset\Dataset.json.').read())
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
        return_list.append({"intent": classes[r[0]], "probability":str(r[1])})
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
    doc = nlp(msg)
    detected_language = doc._.language['language']
    print(f"Detected language chatbot_response:- {detected_language}")

    chatbotResponse = "Loading bot response..."

    if detected_language == 'en':
        res = getResponse(predict_class(msg, model), intents)
        chatbotResponse = res
        print("en_hi chatbot_response", res)
    
    elif detected_language == 'hi':
        translated_msg = translate_text_hi_eng(msg)
        res = getResponse(predict_class(translated_msg, model), intents)
        chatbotResponse = translate_text_eng_hi(res)
        print("hi_en chatbot_response: ", chatbotResponse)

    else:
        msg_in_hindi = hinglish_to_hindi(msg)
        translated_msg = translate_text_hi_eng(msg_in_hindi)
        res = getResponse(predict_class(translated_msg, model), intents)
        chatbotResponse = translate_text_eng_hi(res)
        print("hinglish_hi chatbot_response: ", chatbotResponse)
    
    return chatbotResponse

def translate_text_eng_hi(text):
    translated_text = eng_hi_translator(text, max_length=128, num_beams=5)[0]['generated_text']
    return translated_text

def translate_text_hi_eng(text):
    translated_text = hi_eng_translator(text, max_length=128, num_beams=5)[0]['generated_text']
    return translated_text

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

    doc = nlp(userText)
    detected_language = doc._.language['language']
    
    bot_response_translate = 'Loading bot response...'

    if detected_language == "en":
        userText = spell(userText) 
        bot_response_translate = userText
    
    elif detected_language == 'hi':
        bot_response_translate = translate_text_hi_eng(userText)
        
    else:
        bot_response_translate = hinglish_to_hindi(userText)
    
    chatbot_response_text = chatbot_response(bot_response_translate)

    if detected_language != "en":
         chatbot_response_text = translate_text_eng_hi(chatbot_response_text)

    return chatbot_response_text

if __name__ == "__main__":
    app.run()