
import json 
import torch 
import torch.nn as nn 
import nltk
# from nltk.stem.porter import PorterStemmer
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import ChatROBOT 
from file_handler import tokenize, stem, full_words

nltk.download('punkt') 

with open('intent_software.json', 'r', encoding='utf-8') as f:       ###############################################
     all_questions = json.load(f, strict=False)

all_words, tags, patterns = [], [], []

for intent in all_questions['intents']:
     tag = intent['tag']
     tags.append(tag)

     for question in intent['questions']: 
          word = tokenize(question)               
          all_words.extend(word)
          patterns.append((word, tag))


ignore_characters = ['!', '?', '.', ',']

all_words = [stem(word) for word in all_words if word not in ignore_characters]

all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train, Y_train = [], []

for (sentence_pattern, tag) in patterns:
     bag = full_words(sentence_pattern, all_words)
     X_train.append(bag)
     label = tags.index(tag)
     # print(label)
     Y_train.append(label)

X_train, Y_train = np.array(X_train), np.array(Y_train)


##########################################################################################

class ChatData(Dataset):
     def __init__(self):
         self.n_samples = len(X_train)
         self.x_data    = X_train
         self.y_data    = Y_train

     def __getitem__(self, index):
         return self.x_data[index], self.y_data[index]

     def __len__(self):
          return self.n_samples

##########################################################################################

epochs, batch_size, learning_rate,  = 200, 16, 0.001
input, hidden, output = len(X_train[0]), 32, len(tags)

dataset = ChatData()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ChatROBOT( input, hidden, output).to(device)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for (words, labels) in train_loader:
          
        words  = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        predict = model(words)
        loss = criterion(predict, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'epoch {epoch+1}/{epochs} : Loss :  {loss.item():.6f} ... ')

data = {  'model_state' : model.state_dict(), 'input_size'  : input, 'hidden_size' : hidden,   'output_size' : output,    'all_words'   : all_words,    'tags' : tags }

torch.save(data, 'trained_SOFTWARE')              #####################################################

