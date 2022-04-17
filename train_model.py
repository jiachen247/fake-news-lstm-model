# TAKEN FROM https://www.coursera.org/projects/nlp-fake-news-detector
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
# import keras
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.models import Model

from flask import Flask
from flask import request

app = Flask(__name__)

nltk.download("stopwords")

# load the data
df_true = pd.read_csv("True.csv")
df_fake = pd.read_csv("Fake.csv")

df_true['isfake'] = 1
df_fake['isfake'] = 0
df = pd.concat([df_true, df_fake]).reset_index(drop = True)
df.drop(columns = ['date'], inplace = True)
df['original'] = df['title'] + ' ' + df['text']

# Obtain additional stopwords from nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'reuters'])

# Remove stopwords and remove words with 2 or less characters
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)
            
    return result

# Apply the function to the dataframe
df['clean'] = df['original'].apply(preprocess)

# Obtain the total words present in the dataset
list_of_words = []
for i in df.clean:
    for j in i:
        list_of_words.append(j)

total_words = len(list(set(list_of_words)))

df['clean_joined'] = df['clean'].apply(lambda x: " ".join(x))

# split data into test and train 
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
x_train, x_test, y_train, y_test = train_test_split(df.clean_joined, df.isfake, test_size = 0.2)

# Create a tokenizer to tokenize the words and create sequences of tokenized words
tokenizer = Tokenizer(num_words = total_words)
tokenizer.fit_on_texts(x_train)
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)

print("The encoding for document\n",df.clean_joined[0],"\n is : ",train_sequences[0])

# Add padding can either be maxlen = 4406 or smaller number maxlen = 40 seems to work well based on results
padded_train = pad_sequences(train_sequences,maxlen = 40, padding = 'post', truncating = 'post')
padded_test = pad_sequences(test_sequences,maxlen = 40, truncating = 'post') 

for i, doc in enumerate(padded_train[:2]):
     print("The padded encoding for document",i + 1," is : ", doc)

# Sequential Model
model = Sequential()

# embeddidng layer
model.add(Embedding(total_words, output_dim = 128))
# model.add(Embedding(total_words, output_dim = 240))

# Bi-Directional RNN and LSTM
model.add(Bidirectional(LSTM(128)))

# Dense layers
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1, activation= 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

y_train = np.asarray(y_train)

print("training the model")
model.fit(padded_train, y_train, batch_size = 64, validation_split = 0, epochs = 3)

pred = model.predict(padded_test)

# if the predicted value is >0.5 it is real else it is fake
prediction = []
for i in range(len(pred)):
    if pred[i].item() > 0.5:
        prediction.append(1)
    else:
        prediction.append(0)

print("getting the accuracy")
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(list(y_test), prediction)

print("Model Accuracy : ", accuracy)
print('Model Complete:')

# model.save('saved_model/my_model')

@app.route('/hello', methods=['GET'])
def hello():
    return "Hello there!!"

@app.route('/is_fake', methods=['GET', 'POST'])
def is_fake():
    if request.method == 'POST':
        title = request.form.get('title')
        text = request.form.get('text')
        test_sequence = tokenizer.texts_to_sequences([f"{title} {text}"])
        padded_test = pad_sequences(test_sequence, maxlen = 40, truncating = 'post') 
        return f'Results: {model.predict(padded_test)[0].item()}'

    # otherwise handle the GET request
    return '''
           <form method="POST">
               <div><label>Title: <input type="text" name="title"></label></div>
               <div><label>Text: <input type="textarea" name="text"></label></div>
               <input type="submit" value="Submit">
           </form>'''

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=False, port=5000)
