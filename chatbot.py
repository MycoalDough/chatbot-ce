import random
import json
import pickle
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("wordnet")

import autocorrect as ac

app.config["DEBUG"] = False
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = tf.keras.models.load_model("chatbot_model_001.h5")
print(f"model = {model}")


def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):  # Bag o' words!!!!! bahahahahhahh
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    loi = intents_json["intents"]

    for i in loi:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


@app.route("/predict", methods=["POST"])
def predict():
    # auto correct:
    message = request.json["message"]
    all_words = message.split(" ")
    corrected = []

    for w in all_words:
        corrected.append(ac.auto_correct(w, words))

    new_message = ac.combine_to_list(corrected)

    ints = predict_class(new_message)
    res = get_response(ints, intents)
    return jsonify({"response": res})


if __name__ == "__main__":
    app.run(debug=False)
