import random
import json 
import torch 
from file_handler import  full_words, tokenize
from model import ChatROBOT

# with open('flask.json', 'r') as f:
#      all_questions = json.load(f)

# nltk.download('punkt') 

with open('data1_flask.json', 'r', encoding='utf-8') as f:
     all_questions = json.load(f, strict=False)

trained_model = torch.load('trained_ChatROBOT')


model_state = trained_model['model_state']
input_size  = trained_model['input_size']
hidden_size = trained_model['hidden_size']
output_size = trained_model['output_size']
all_words   = trained_model['all_words']
tags        = trained_model['tags'] 

model = ChatROBOT(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval 

print()
bot_name, your_name = "chatBOT", "You"

# print("\nWelcome", your_name.upper())

while True:
    sentence = input(f"{your_name.title():15}   :     ")
    if sentence == "exit":
        print("Thanks for joining me!")
        break

    sentence = tokenize(sentence)
    X = full_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # print(prob)
    if prob.item() > 0.75:
        for intent in all_questions['intents']:
            if tag == intent['tag']:
                print(f"{bot_name:15}   :     {random.choice(intent['answers'])}")
    else:
        print(f"{bot_name:15}   :     Be more specific ")


        