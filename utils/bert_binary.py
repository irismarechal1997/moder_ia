#import all packages
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertConfig, AutoTokenizer, TFBertModel, BertTokenizer, TFBertForSequenceClassification, BertModel
#import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def bert_model_1(processed=False):

    filepath="weights-bert.h5"
    checkpoint_callback = ModelCheckpoint(
    filepath=filepath,  # Specify the path to save the checkpoint file
    save_best_only=True,  # Save only the best model (based on validation loss)
    monitor='val_loss',  # Metric to monitor for saving the best model
    mode='min',  # In this case, we're monitoring for the minimum validation loss
    verbose=1)  # Display progress while saving)

    #quick cleaning
    data=pd.read_csv('data/processed_dataset_v1.csv')
    data['text_processed'] = data['text_processed'].astype(str)
    data['text_processed'] = data['text_processed'].str.strip()

    # Prepare X and y
    X = data['text_processed']
    y = data['offensive']

    # Création des test et des trains
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

    X_train=X_train.tolist()
    X_test = X_test.tolist()

    # Initializing a BERT mini model style configuration

    config = BertConfig.from_pretrained('prajjwal1/bert-mini')
    #config.hidden_size = 256
    #config.num_hidden_layers = 6
    tokenizer_mini = BertTokenizer.from_pretrained('prajjwal1/bert-mini')
    model_mini = TFBertForSequenceClassification.from_pretrained('prajjwal1/bert-mini',from_pt=True, config=config)
    optimizer=keras.optimizers.Adam(learning_rate=0.0001)
    model_mini.compile(optimizer=optimizer, metrics=["accuracy"])

    #reduction des paramètres
    model_mini.bert.trainable = False #to be switched to False if enough space/memory


    #tokenisation
    X_train_tokenized = tokenizer_mini(X_train, return_tensors='tf', padding=True, truncation=True, max_length=512)
    X_test_tokenized = tokenizer_mini(X_test, return_tensors='tf', padding=True, truncation=True, max_length=512)

    X_train_tokenized = dict(X_train_tokenized)
    X_test_tokenized = dict(X_test_tokenized)
    es=EarlyStopping(patience=2, restore_best_weights=True, monitor='loss')

    model_mini.fit(X_train_tokenized,y_train, batch_size=32, epochs=10, callbacks=[es,checkpoint_callback],validation_split=.2)

    res = model_mini.evaluate(X_test_tokenized, y_test)

    print(f'The accuracy evaluated on the test set is of {res[1]*100:.3f}%')

    return model_mini
