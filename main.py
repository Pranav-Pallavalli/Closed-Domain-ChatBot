from re import S
import nltk
from nltk import tag
from nltk.inference import resolution
from nltk.stem.lancaster import LancasterStemmer
import random
import json
from nltk.util import pr
import tensorflow
import tflearn
import numpy
import pickle
stemmer = LancasterStemmer()
with open("intents.json") as f:
    data = json.load(f)
try:
    with open("data.pickle","rb") as f:
        words,labels,training,output = pickle.load(f)
except:
    words = []
    docs_a = []
    docs_b = []
    labels = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            x = nltk.word_tokenize(pattern)
            words.extend(x)
            docs_a.append(x)
            docs_b.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w!="?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    output_empty = [0 for _ in range(len(labels))]
    for i,doc in enumerate(docs_a):
        bagOfW = []
        x = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            if w in x:
                bagOfW.append(1)
            else:
                bagOfW.append(0)
        output_row = output_empty[:]
        output_row[labels.index(docs_b[i])] = 1
        training.append(bagOfW)
        output.append(output_row)
        with open("data.pickle","wb") as f:
            pickle.dump((words,labels,training,output),f)

training = numpy.array(training)
output = numpy.array(output)
net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]),activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)
try:
    model.load("model.tflearn")
except:
    model.fit(training,output,n_epoch=1000,batch_size=8,show_metric=True)
    model.save("model.tflearn")


def bag_of_words(sent,words):
    bag = [0 for _ in range(len(words))]
    s = nltk.word_tokenize(sent)
    s = [stemmer.stem(i) for i in s]
    for i in s:
        for j,w in enumerate(words):
            if w==s:
                bag[j]=1
    return numpy.array(bag)


def Chat():
    inp = input("CHATBOT: "+"Hey,how's it goin?,what can i help you with today?(Type quit to stop talking to the chatBot)\n")
    while True:
        if inp==quit:
            break
        result = model.predict(bag_of_words(inp,words))
        resultInd = numpy.argmax(result)
        tag = labels[resultInd]
        for t in data["intents"]:
            if t["tag"]== tag:
                response = t["responses"]
        print("ChatBot "+str(random.choice(response))+"\n")

Chat()


# print(words)
# print(docs_a)
# print(docs_b)
# print(labels)