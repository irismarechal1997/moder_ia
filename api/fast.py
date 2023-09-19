#import general packages
import pandas as pd

#import functions
from utils.registry import load_model
from utils.data_preproc import cleaning_text

#import fast_api
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

#import packages
from transformers import  BertTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import Tokenizer
from transformers import BertConfig, AutoTokenizer, TFBertModel, BertTokenizer, TFBertForSequenceClassification, BertModel


app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    )

config = BertConfig.from_pretrained('prajjwal1/bert-mini')
model = TFBertForSequenceClassification(config=config)
model.build()
model.load_weights("weights-bert.h5")
print("✅ 0. model loaded")

# $WIPE_END

@app.get("/predict")
def predict_binary(X_pred: str):

    #preprocessing
    X_pred = str(X_pred)
    X_pred = cleaning_text(X_pred)
    print("✅ 1. data cleaned")

    #tokenizer_bert
    tokenizer_mini = BertTokenizer.from_pretrained('prajjwal1/bert-mini')
    X_pred = tokenizer_mini(X_pred, return_tensors='tf', padding=True, truncation=True, max_length=512)
    print("✅ 2. data tokenized")

    model = model

    y_pred = model.predict(X_pred)
    print("✅ 3. prediction made")
    print(y_pred)

    if y_pred == 1:
        prediction = "❌ offensive_tweet"
    else:
        prediction = "✅ non-offensive_tweet"

    print(f'type of tweet: {prediction}')

    return {"type of tweet": prediction}



# @app.get("/")
# def root():
#     # $CHA_BEGIN
#     return dict(greeting="Hello")
#     # $CHA_END
