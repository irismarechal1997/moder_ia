
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
#import tensorflow as tf
#import keras
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding


# import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVC
import random



import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, hamming_loss



# Machine Learning models for multi label classification

def classif_GRU_model():

    data_processed = pd.read_csv("data/"+"labelling_dataset_v1.csv")
    # Split into training and testing data
    X = data_processed["text_processed"]
    y = data_processed.drop(labels=["text_processed", "text"], axis=1)
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


    # Create the model
    model = Sequential()
    model.add(layers.Embedding(
        input_dim=vocab_size+1, # size of the input, impacting the number of weights in the linear combinations of the neurons of the first layer
        output_dim=embedding_size, # 100
        mask_zero=True, # Built-in masking layer
    ))

    model.add(layers.GRU(20, return_sequences=True, activation="tanh"))
    model.add(layers.GRU(20, activation="tanh"))
    model.add(layers.Dense(6, activation="sigmoid"))
    model.summary()

    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    es = EarlyStopping(patience=4, restore_best_weights=True)

    model.fit(X_train_pad, y_train,
            epochs=150,
            batch_size=128,
            callbacks=[es], validation_split=0.3
            )

    res = model.evaluate(X_test_pad, y_test)
    print(f'The accuracy evaluated on the test set is of {res[1]*100:.3f}%')
    print('Testing loss \t', res[0]*100)
    print('Testing accuracy ', res[1]*100)
    return model



def classif_cnn_model():
    data_processed = pd.read_csv("data/"+"labelling_dataset_v1.csv")
    # Split into training and testing data
    X = data_processed["text_processed"]
    y = data_processed.drop(labels=["text_processed", "text"], axis=1)
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

    # create the model
    model = Sequential()

    model.add(Embedding(input_dim=vocab_size+1, output_dim=embedding_size))
    model.add(layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.LSTM(100))
    model.add(layers.Dense(6, activation='sigmoid'))

    # Students will be ending their code here

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Change the number of epochs and the batch size depending on the RAM Size

    es = EarlyStopping(patience=4, restore_best_weights=True)

    model.fit(X_train_pad, y_train,
            epochs=150,
            batch_size=128,
            callbacks=[es], validation_split=0.3
            )
    res = model.evaluate(X_test_pad, y_test)
    print(f'The accuracy evaluated on the test set is of {res[1]*100:.3f}%')
    print('Testing loss \t', res[0]*100)
    print('Testing accuracy ', res[1]*100)

    return model


def full_model_classif():

    labelled_data = pd.read_csv("/home/mariannettrd/code/irismarechal1997/moder_ia/data/labelling_dataset_v1.csv")
    label = np.zeros(56515, dtype=np.int8 )
    list_1 = label.tolist()
    labelled_data.insert(1,"non-offensive",list_1)

    df = pd.read_csv("/home/mariannettrd/code/irismarechal1997/moder_ia/data/processed_dataset_v1.csv")


    #select number of tweets
    number_rows = 10_000

    #filter on non-offensive
    df2 = df[df["offensive"]==0]
    sample_df=df2.sample(number_rows)

    #select right columns to keep
    selected_features = ["text","offensive", "text_processed"]
    new_df = sample_df[selected_features].copy()
    label = np.zeros(number_rows, dtype=np.int8 )
    list_2 = label.tolist()

    # Create a new DataFrame based on sample_df
    new_df = sample_df[selected_features].copy()

    # Define the columns you want to insert
    columns= ['homophobia', 'transphobia', 'misogyny', 'xenophobia', 'religion', 'racism', 'non-offensive']

    # Iterate through the columns and insert them into new_df
    for col in columns:
        new_df.insert(1, col, list_2)
        new_df["non-offensive"]=1
        new_df.drop(columns="offensive", inplace = True)

    data_set= pd.concat([labelled_data, new_df])

    # Split into training and testing data
    X = data_set["text"]
    y = data_set.drop(labels=["text_processed", "text"], axis=1)
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


    # Create the model
    model = Sequential()
    model.add(layers.Embedding(
        input_dim=vocab_size+1, # size of the input, impacting the number of weights in the linear combinations of the neurons of the first layer
        output_dim=embedding_size, # 100
        mask_zero=True, # Built-in masking layer
    ))

    model.add(layers.GRU(20, return_sequences=True, activation="tanh"))
    model.add(layers.GRU(20, activation="tanh"))
    model.add(layers.Dense(7, activation="sigmoid"))
    model.summary()

    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])


    es = EarlyStopping(patience=10, restore_best_weights=True)

    model.fit(X_train_pad, y_train,
            epochs=150,
            batch_size=128,
            callbacks=[es], validation_split=0.3
            )

    res = model.evaluate(X_test_pad, y_test)
    print(f'The accuracy evaluated on the test set is of {res[1]*100:.3f}%')

    return model
