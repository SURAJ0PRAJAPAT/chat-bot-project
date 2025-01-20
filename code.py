import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import nltk
from nltk.corpus import wordnet
from googletrans import Translator
import langid
from textblob import TextBlob
import spacy
import logging
import random

nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")

logging.basicConfig(filename='chatbot.log', level=logging.INFO)

data_path = '/kaggle/input/bitext-gen-ai-chatbot-customer-support-dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv'
data = pd.read_csv(data_path)

print(data.head())
print(data.info())

data = data.dropna(subset=['instruction', 'intent', 'response'])
data['combined'] = data['instruction'] + " " + data['response']

X = data['instruction']
y = data['intent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
print(classification_report(y_test, predictions))

intent_counts = data['intent'].value_counts()
plt.figure(figsize=(10, 6))
intent_counts.plot(kind='bar', color='skyblue')
plt.title('Intent Distribution')
plt.xlabel('Intent')
plt.ylabel('Count')
plt.show()

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)
with open('intent_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

def synonym_replacement(text, n=2):
    words = text.split()
    new_words = words.copy()
    for _ in range(n):
        word = random.choice(words)
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = random.choice(synonyms.lemmas()).name()
            new_words = [synonym if w == word else w for w in new_words]
    return ' '.join(new_words)

def translate_text(text, target_language="en"):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text

def detect_language(text):
    return langid.classify(text)[0]

def extract_entities(text):
    doc = nlp(text)
    return {ent.text: ent.label_ for ent in doc.ents}

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def log_conversation(user_input, response):
    logging.info(f"User Input: {user_input}, Response: {response}")

def generate_response(user_input):
    user_input_tfidf = vectorizer.transform([user_input])
    intent = model.predict(user_input_tfidf)[0]
    confidence = model.predict_proba(user_input_tfidf).max()
    if confidence < 0.5:
        return "I'm sorry, I didn't understand that. Can you please rephrase?"
    else:
        return f"Detected intent: {intent}. How can I assist you further?"

def enhanced_chatbot(user_input, target_language="en"):
    detected_lang = detect_language(user_input)
    if detected_lang != "en":
        user_input_en = translate_text(user_input, target_language="en")
    else:
        user_input_en = user_input
    entities = extract_entities(user_input_en)
    sentiment = analyze_sentiment(user_input_en)
    response_en = generate_response(user_input_en)
    if target_language != "en":
        response = translate_text(response_en, target_language=target_language)
    else:
        response = response_en
    log_conversation(user_input, response)
    return response

user_input = "¿Cómo cancelo mi pedido?"
response = enhanced_chatbot(user_input, target_language="es")
print(response)

user_input = "Quelle est la politique de remboursement ?"
response = enhanced_chatbot(user_input, target_language="fr")
print(response)

user_input = "How do I track my order?"
response = enhanced_chatbot(user_input, target_language="en")
print(response)
