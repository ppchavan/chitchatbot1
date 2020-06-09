import argparse
import nltk
import os
from nltk.stem.lancaster import LancasterStemmer
nltk.download('punkt')
stemmer = LancasterStemmer()
import json
import pickle
import random
import numpy
import tflearn as tf
import tensorflow

MODEL_NAME = "model.tflearn"

with open('intent.json') as file:
    data = json.load(file)

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat(model, words, labels):
    print("Start talking with the bot (type quit or q to stop chatting.) \n Have fun!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit" or inp.lower() == 'q':
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        full_label = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] + tg['persona'] == full_label:
                responses = tg['responses']

        print(random.choice(responses))

def main(args):
    print("Welcome to chitchat bot.")

    try:
        with open("data.pickle", "rb") as f:
            words, labels, training, output = pickle.load(f)
    except:
        words = []
        labels = []
        docs_x = []
        docs_y = []

        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                label = intent["tag"] + intent['persona']
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(label)

            if label not in labels:
                labels.append(label)

        words = [stemmer.stem(w.lower()) for w in words if w != "?"]
        words = sorted(list(set(words)))

        labels = sorted(labels)

        training = []
        output = []

        out_empty = [0 for _ in range(len(labels))]

        for x, doc in enumerate(docs_x):
            bag = []

            wrds = [stemmer.stem(w.lower()) for w in doc]

            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1

            training.append(bag)
            output.append(output_row)

        training = numpy.array(training)
        output = numpy.array(output)

        with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)

    tensorflow.reset_default_graph()

    net = tf.input_data(shape=[None, len(training[0])])
    net = tf.fully_connected(net, 8)
    net = tf.fully_connected(net, 8)
    net = tf.fully_connected(net, len(output[0]), activation="softmax")
    net = tf.regression(net)

    model = tf.DNN(net)
    
    # Load existing model if one exists. Otherwise save one and then use it.  
    if os.path.exists(MODEL_NAME + ".meta"):
        model.load(MODEL_NAME)
    else:
        model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
        model.save(MODEL_NAME)

    # Main chat program
    chat(model, words, labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)