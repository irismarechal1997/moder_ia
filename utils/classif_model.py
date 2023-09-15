from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
#import tensorflow as tf
#import keras
from tensorflow.keras.utils import pad_sequences

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
# import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping


# Baseline multi-label classification

def LSTM_model():
    data_processed = pd.read_csv("data/"+"processed_dataset_v1.csv")
    X = xx
    y = xx
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)

    ### Let's tokenize the vocabulary
    tk = Tokenizer()
    tk.fit_on_texts(X_train)
    vocab_size = len(tk.word_index)

    # We apply the tokenization to the train and test set
    X_train_token = tk.texts_to_sequences(X_train)
    X_test_token = tk.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_token, dtype='float32', padding='post')
    X_test_pad = pad_sequences(X_test_token, dtype='float32', padding='post')

    deep_inputs = Input(shape=(maxlen,))
    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)
    LSTM_Layer_1 = LSTM(128)(embedding_layer)
    dense_layer_1 = Dense(6, activation='sigmoid')(LSTM_Layer_1)
    model = Model(inputs=deep_inputs, outputs=dense_layer_1)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
