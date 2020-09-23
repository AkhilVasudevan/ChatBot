import nltk

import numpy as np

#stemming technique (porter)
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

#tokenizing lib of nltk
#nltk.download('punkt')

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence,all_words):
    tokenized_sentence=[stem(w) for w in tokenized_sentence]
    #converting to 0,1 array
    bag=np.zeros(len(all_words),dtype=np.float32)
    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx]=1.0
    return bag