#import neccessary stuff
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import os
import numpy
import tflearn
import tensorflow
import random
import json
import time


with open("intents.json") as file:
    data=json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

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

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except ValueError:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

from selenium import webdriver
import time
import pickle

#Open Whatsapp with WebDriver and manually lohin from your mobile phone
whatsapp = webdriver.Chrome('chromedriver.exe') 
whatsapp.get('https://web.whatsapp.com/')

#Define a flag to check repeated messages
flag='NULLL'

message_box=''#Div class of message box
send_button=''#div class of send button
in_message=''#div class of incoming message
def send():
    results = model.predict([bag_of_words(message, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
    reply=random.choice(responses)
    box=whatsapp.find_element_by_class_name(message_box)
    box.click()
    box.send_keys(reply)
    send=whatsapp.find_element_by_class_name(send_button)
    send.click()

while True:
    #Identify last sent message
    message=whatsapp.find_elements_by_xpath('//div[@class="{}"]'.format(in_message))[-1].text.split('\n')[0]
    if message==flag:
        continue
        time.sleep(1)
    elif message=="Quit":
        break
    else:
        send()
        flag=message
