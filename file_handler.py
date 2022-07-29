
import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np



def tokenize(sentence):
     return nltk.word_tokenize(sentence)

stemmer = PorterStemmer()
def stem(sentence):
     return stemmer.stem(sentence.lower())

def full_words(tokenized, full_words):
     complete_words = [stem(word) for word in tokenized]
     bag = np.zeros(len(full_words), dtype=np.float32)
     for index, w in enumerate(full_words):
          if w in complete_words:
               bag[index] = 1
     return bag 