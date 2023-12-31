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



def LSTM_model(processed=True):

    data_processed = pd.read_csv("data/"+"processed_dataset_v1.csv")

    if processed :
        X = data_processed["text"]
        y = data_processed["offensive"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)
    else :
        X = data_processed["text_processed"]
        y = data_processed["offensive"]
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

    ### Let's build the neural network now

    # Size of your embedding space = size of the vector representing each word
    embedding_size = 50

    model = Sequential()
    model.add(layers.Embedding(
        input_dim=vocab_size+1, # size of the input, impacting the number of weights in the linear combinations of the neurons of the first layer
        output_dim=embedding_size, # 100
        mask_zero=True, # Built-in masking layer
    ))

    model.add(layers.LSTM(20, return_sequences=True, activation="tanh"))
    model.add(layers.LSTM(20, activation="tanh"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()

    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])


    es = EarlyStopping(patience=4, restore_best_weights=True)

    model.fit(X_train_pad, y_train,
            epochs=5,
            batch_size=128,
            callbacks=[es]
            )
    res = model.evaluate(X_test_pad, y_test)
    print(f'The accuracy evaluated on the test set is of {res[1]*100:.3f}%')
    return model


def GRU_model(processed=True):

    data_processed = pd.read_csv("data/"+"processed_dataset_v1.csv")

    if processed :
        X = data_processed["text"]
        y = data_processed["offensive"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)
    else :
        X = data_processed["text_processed"]
        y = data_processed["offensive"]
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

    # Size of your embedding space = size of the vector representing each word
    embedding_size = 50

    model = Sequential()
    model.add(layers.Embedding(
        input_dim=vocab_size+1, # size of the input, impacting the number of weights in the linear combinations of the neurons of the first layer
        output_dim=embedding_size, # 100
        mask_zero=True, # Built-in masking layer
    ))

    model.add(layers.GRU(20, return_sequences=True, activation="tanh"))
    model.add(layers.GRU(20, activation="tanh"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()

    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])


    es = EarlyStopping(patience=4, restore_best_weights=True)

    model.fit(X_train_pad, y_train,
            epochs=5,
            batch_size=128,
            callbacks=[es]
            )

    res = model.evaluate(X_test_pad, y_test)
    print(f'The accuracy evaluated on the test set is of {res[1]*100:.3f}%')
    return model


def Conv1D_model(processed=True):

    data_processed = pd.read_csv("data/"+"processed_dataset_v1.csv")

    if processed :
        X = data_processed["text"]
        y = data_processed["offensive"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)
    else :
        X = data_processed["text_processed"]
        y = data_processed["offensive"]
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

    ### Let's build the neural network now

    # Size of your embedding space = size of the vector representing each word
    embedding_size = 50

    model = Sequential()
    model.add(layers.Embedding(
        input_dim=vocab_size+1, # size of the input, impacting the number of weights in the linear combinations of the neurons of the first layer
        output_dim=embedding_size, # 100
        mask_zero=True, # Built-in masking layer
    ))

    model.add(layers.Conv1D(1, 20, activation="tanh"))
    model.add(layers.Conv1D(1, 20,activation="tanh"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()

    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])


    es = EarlyStopping(patience=4, restore_best_weights=True)

    model.fit(X_train_pad, y_train,
            epochs=5,
            batch_size=128,
            callbacks=[es]
            )

    res = model.evaluate(X_test_pad, y_test)

    print(f'The accuracy evaluated on the test set is of {res[1]*100:.3f}%')

    return model
