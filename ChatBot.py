import random
import json

import torch
import torch.nn as nn

from nltk_utilities import bag_of_words, tokenize
from model import *

def ChatBot(sentence):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('./training_data/train.json', 'r') as json_data:
        intents = json.load(json_data)
    FILE = "data.pth"
    data = torch.load(FILE)
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['train']:
            if tag == intent["tag"]:
                return "{0}".format(random.choice(intent['responses']))
    else:
        return "I do not understand..."