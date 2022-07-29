from urllib import response
from flask import Flask, render_template, request 
from file_handler import  full_words, tokenize
from model import ChatROBOT
import torch 
import json 
import random


app = Flask(__name__)

###########################################################################################
###########################################################################################


@app.route('/')
def home():   
    return render_template('home.html')


###########################################################################################
###########################################################################################

chat_bootstrap = {}
@app.route('/bootstrap', methods=['GET', 'POST'])                               #############
def bootstrap(): 
    response = None 

    if request.method == 'POST':
        user_input =  request.form['question_bootstrap']                       #############

        with open('intent_bootstrap.json', 'r', encoding='utf-8') as f:        #############
            all_questions = json.load(f, strict=False)

        trained_model = torch.load('trained_BOOTSTRAP')                        #############

        model_state = trained_model['model_state']
        input_size  = trained_model['input_size']
        hidden_size = trained_model['hidden_size']
        output_size = trained_model['output_size']
        all_words   = trained_model['all_words']
        tags        = trained_model['tags'] 

        model = ChatROBOT(input_size, hidden_size, output_size)
        model.load_state_dict(model_state)
        model.eval 

        question = tokenize(user_input)
        X = full_words(question, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in all_questions['intents']:
                if tag == intent['tag']:
                    response = random.choice(intent['answers'])
        else:
            response = 'Be more specific'
        
        chat_bootstrap[user_input] = response

    return render_template('chatbot_bootstrap.html',  chat = chat_bootstrap)       #############

###########################################################################################
###########################################################################################

chat_css = {}
@app.route('/css', methods=['GET', 'POST'])                             #############
def css():   
    response = None 

    if request.method == 'POST':
        user_input =  request.form['question_css']                      #############

        with open('intent_css.json', 'r', encoding='utf-8') as f:       #############
            all_questions = json.load(f, strict=False)

        trained_model = torch.load('trained_CSS')                       #############

        model_state = trained_model['model_state']
        input_size  = trained_model['input_size']
        hidden_size = trained_model['hidden_size']
        output_size = trained_model['output_size']
        all_words   = trained_model['all_words']
        tags        = trained_model['tags'] 

        model = ChatROBOT(input_size, hidden_size, output_size)
        model.load_state_dict(model_state)
        model.eval 

        question = tokenize(user_input)
        X = full_words(question, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in all_questions['intents']:
                if tag == intent['tag']:
                    response = random.choice(intent['answers'])
        else:
            response = 'Be more specific'
        
        chat_css[user_input] = response                                 #############

    return render_template('chatbot_css.html', chat = chat_css)         #############

###########################################################################################
###########################################################################################

chat_docker = {}
@app.route('/docker', methods=['GET', 'POST'])                              #############
def docker(): 
    response = None 

    if request.method == 'POST':
        user_input =  request.form['question_docker']                       #############

        with open('intent_docker.json', 'r', encoding='utf-8') as f:        ##############
            all_questions = json.load(f, strict=False)

        trained_model = torch.load('trained_DOCKER')                        #############

        model_state = trained_model['model_state']
        input_size  = trained_model['input_size']
        hidden_size = trained_model['hidden_size']
        output_size = trained_model['output_size']
        all_words   = trained_model['all_words']
        tags        = trained_model['tags'] 

        model = ChatROBOT(input_size, hidden_size, output_size)
        model.load_state_dict(model_state)
        model.eval 

        question = tokenize(user_input)
        X = full_words(question, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in all_questions['intents']:
                if tag == intent['tag']:
                    response = random.choice(intent['answers'])
        else:
            response = 'Be more specific'
        
        chat_docker[user_input] = response                              #############

    return render_template('chatbot_docker.html', chat = chat_docker)   #############

#############################################################################################
#############################################################################################

chat_flask = {}
@app.route('/flask', methods=['GET', 'POST'])                             #############
def flask():   
    response = None 

    if request.method == 'POST':
        user_input =  request.form['question_flask']                      #############

        with open('intent_flask.json', 'r', encoding='utf-8') as f:       ###############
            all_questions = json.load(f, strict=False)

        trained_model = torch.load('trained_FLASK')                        #############

        model_state = trained_model['model_state']
        input_size  = trained_model['input_size']
        hidden_size = trained_model['hidden_size']
        output_size = trained_model['output_size']
        all_words   = trained_model['all_words']
        tags        = trained_model['tags'] 

        model = ChatROBOT(input_size, hidden_size, output_size)
        model.load_state_dict(model_state)
        model.eval 

        question = tokenize(user_input)
        X = full_words(question, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in all_questions['intents']:
                if tag == intent['tag']:
                    response = random.choice(intent['answers'])         #############
        else:
            response = 'Be more specific'
        
        chat_flask[user_input] = response                                #############

    return render_template('chatbot_flask.html', chat = chat_flask)


#################################################################################################
#################################################################################################

chat_general = {}
@app.route('/general', methods=['GET', 'POST'])                             #############
def general(): 
    response = None 

    if request.method == 'POST':
        user_input =  request.form['question_general']                      #############

        with open('intent_general.json', 'r', encoding='utf-8') as f:       #############
            all_questions = json.load(f, strict=False)

        trained_model = torch.load('trained_GENERAL')                       #############

        model_state = trained_model['model_state']
        input_size  = trained_model['input_size']
        hidden_size = trained_model['hidden_size']
        output_size = trained_model['output_size']
        all_words   = trained_model['all_words']
        tags        = trained_model['tags'] 

        model = ChatROBOT(input_size, hidden_size, output_size)
        model.load_state_dict(model_state)
        model.eval 

        question = tokenize(user_input)
        X = full_words(question, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in all_questions['intents']:
                if tag == intent['tag']:
                    response = random.choice(intent['answers'])
        else:
            response = 'Be more specific'
        
        chat_general[user_input] = response

    return render_template('chatbot_general.html',  chat = chat_general)       #############


###################################################################################################
###################################################################################################

chat_html = {}
@app.route('/html', methods=['GET', 'POST'])                                #############
def html():  
    response = None 

    if request.method == 'POST':
        user_input =  request.form['question_html']                         #############

        with open('intent_html.json', 'r', encoding='utf-8') as f:          ###############
            all_questions = json.load(f, strict=False)

        trained_model = torch.load('trained_HTML')                          #############

        model_state = trained_model['model_state']
        input_size  = trained_model['input_size']
        hidden_size = trained_model['hidden_size']
        output_size = trained_model['output_size']
        all_words   = trained_model['all_words']
        tags        = trained_model['tags'] 

        model = ChatROBOT(input_size, hidden_size, output_size)
        model.load_state_dict(model_state)
        model.eval 

        question = tokenize(user_input)
        X = full_words(question, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in all_questions['intents']:
                if tag == intent['tag']:
                    response = random.choice(intent['answers'])
        else:
            response = 'Be more specific'
        
        chat_html[user_input] = response                                #############


    return render_template('chatbot_html.html',  chat = chat_html)      #############



###################################################################################################
###################################################################################################

chat_software = {}
@app.route('/software', methods=['GET', 'POST'])                            #############
def software(): 
    response = None 

    if request.method == 'POST':
        user_input =  request.form['question_software']                     #############

        with open('intent_software.json', 'r', encoding='utf-8') as f:       ############
            all_questions = json.load(f, strict=False)

        trained_model = torch.load('trained_SOFTWARE')                      #############

        model_state = trained_model['model_state']
        input_size  = trained_model['input_size']
        hidden_size = trained_model['hidden_size']
        output_size = trained_model['output_size']
        all_words   = trained_model['all_words']
        tags        = trained_model['tags'] 

        model = ChatROBOT(input_size, hidden_size, output_size)
        model.load_state_dict(model_state)
        model.eval 

        question = tokenize(user_input)
        X = full_words(question, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in all_questions['intents']:
                if tag == intent['tag']:
                    response = random.choice(intent['answers'])
        else:
            response = 'Be more specific'
        
        chat_software[user_input] = response                                    #############

    return render_template('chatbot_software.html', chat = chat_software)       #############

#####################################################################################################
#####################################################################################################

if __name__ == '__main__':
    app.run(debug=True)