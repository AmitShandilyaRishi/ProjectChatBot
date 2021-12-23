import pickle
from django.shortcuts import render
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
# import tensorflow
from tensorflow.python.framework import ops
import random
import json


# Create your views here.
def chat(request):
    you = ""
    if request.method == "POST":
        message = request.POST.get('message')
        you = message
        print("You : ", end=' ')
        print(you)

        stemmer = LancasterStemmer()
        with open("actions.json") as file:
            data = json.load(file)
            #print(data)
        try:
            with open("data.pickle", "rb") as f:
                key_words, category_labels, training, output = pickle.load(f)

        except:
            key_words = []
            category_labels = []
            documents_x = []
            documents_y = []

            for action in data["actions"]:
                for question in action["questions"]:
                    wrds = nltk.word_tokenize(question)
                    key_words.append(wrds)
                    documents_x.append(wrds)
                    documents_y.append(action["category"])

                    if action["category"] not in category_labels:
                        category_labels.append(action["category"])

            key_words = [stemmer.stem(w.lower()) for w in key_words if w != "?"]
            key_words = sorted(list(set(key_words)))
            category_labels = sorted(category_labels)

            training = []
            output = []

            out_empty = [0 for _ in range(len(category_labels))]

            for x, doc in enumerate(documents_x):
                bag = []

                wrds = [stemmer.stem(w.lower()) for w in doc]

                for w in key_words:
                    if w in wrds:
                        bag.append(1)
                    else:
                        bag.append(0)

                output_row = out_empty[:]
                output_row[category_labels.index(documents_y[x])] = 1

                training.append(bag)
                output = numpy.array(output)

            training = numpy.array(training)
            output = numpy.array(output)

            with open("data.pickle", "wb") as f:
                pickle.dump((key_words, category_labels, training, output), f)

        ops.reset_default_graph()

        net = tflearn.input_data(shape=[None, len(training[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
        net = tflearn.regression(net)

        model = tflearn.DNN(net)

        try:
            model.load('model.tflearn')
        except:
            ops.reset_default_graph()

            net = tflearn.input_data(shape=[None, len(training[0])])
            net = tflearn.fully_connected(net, 8)
            net = tflearn.fully_connected(net, 8)
            net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
            net = tflearn.regression(net)

            model = tflearn.DNN(net)
            model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
            model.save('model.tflearn')

        def bag_of_words(s, key_words):
            bag = [0 for _ in range(len(key_words))]

            s_words = nltk.word_tokenize(s)
            s_words = [stemmer.stem(word.lower()) for word in s_words]

            for se in s_words:
                for i, w in enumerate(key_words):
                    if w == se:
                        bag[i] = 1

            return numpy.array(bag)

        def chat1():

            results = model.predict([bag_of_words(you, key_words)])[0]
            results_index = numpy.argmax(results)
            tag = category_labels[results_index]

            if results[results_index] > 0.7:
                for tg in data["actions"]:
                    if tg['category'] == tag:
                        responses = tg['responses']
                print("Bot ! : ", end=' ')
                print(random.choice(responses))
            else:
                print("Bot ! : ", end=' ')
                print("I didn't get that, try again.")
        chat1()

    return render(request, 'chatbot/index.html', {"your_message": you})
