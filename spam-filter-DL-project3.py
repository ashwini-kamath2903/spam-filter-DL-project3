import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Dropout
from tensorflow.keras.models import Model

# Loading the training data
train = pd.read_csv('/kaggle/input/spam-train-csv/train.csv')

train.head()

# Reading word embeddings from a GloVe file and storing them in a dictionary
embedding_index = {}
f = open(r"/kaggle/input/glove-dataset/glove.6B.100d.txt", encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    wts = np.asarray(values[-100:], dtype='float32')
    embedding_index[word] = wts
f.close()

# Word vectors from GloVe file associated with 'today'
embedding_index['today']

from sklearn.model_selection import train_test_split
Train_train, Test_test=train_test_split(train, test_size=0.2)

x_train=Train_train['question_text']
y_train=Train_train['target']
x_test=Test_test['question_text']
y_test=Test_test['target']

# New column to dataframe 'train' with name:'length'
train['length']=[len(x) for x in train['question_text']]

# To get the maximum and minimum values in the 'length' column of the DataFrame train
train['length'].max(),train['length'].min()

max_len=251
# To create a Tokenizer object named 'Tk' for tokenizing text data.
Tk=Tokenizer(char_level=False,split=' ')
# Tokenizer learns the vocabulary and tokenizes the text in x_train
Tk.fit_on_texts(x_train)

# To calculate the number of unique words (or tokens) in your text data, which is the vocabulary size of the tokenizer('Tk')
vocab_size=len(Tk.word_index)

vocab_size

# To convert the text sequences in your training dataset x_train into sequences of integers
seq_train=Tk.texts_to_sequences(x_train)
# To convert the text sequences in your test dataset x_test into a matrix format.
seq_test=Tk.texts_to_matrix(x_test)

# Padding the x_train and x_test files to the max. length taken
seq_train_matrix=sequence.pad_sequences(seq_train,maxlen=max_len)
seq_test_matrix=sequence.pad_sequences(seq_test,maxlen=max_len)

# To initialize the embedding layer in a deep learning model with pre-trained word embeddings.
embedding_matrix=np.zeros((vocab_size+1,100))

# To fill the embedding_matrix with pre-trained word vectors obtained from the embedding_index dictionary
for word,i in Tk.word_index.items():
  # To retrieve the pre-trained word vector for the current word word from the embedding_index dictionary.
  embed_vector=embedding_index.get(word)

  if embed_vector is not None:
    embedding_matrix[i]=embed_vector

# Define input and embedding layers
inputs = Input(name='question_text', shape=[max_len])
embed = Embedding(vocab_size + 1, 100, input_length=max_len, mask_zero=True, weights=[embedding_matrix], trainable=False)(inputs)

# Define the LSTM layer with unroll=True to disable cuDNN
lstm_layer = LSTM(50, recurrent_activation='sigmoid', unroll=True)(embed)

# Define remaining layers
dense1 = Dense(10, activation='relu')(lstm_layer)
dr1 = Dropout(0.2)(dense1)
final_layer = Dense(1, activation='sigmoid')(dr1)

# Build and compile the model
model = Model(inputs=inputs, outputs=final_layer)
model.summary()

# Compiling and training the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(seq_train_matrix, y_train, validation_data=[seq_test_matrix, y_test], epochs=20, batch_size=50)
