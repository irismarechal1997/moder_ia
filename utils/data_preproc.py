import pandas as pd
import numpy as np
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')


def cleaning_table(data):
    data = data.drop_duplicates() # Remove duplicates
    data = data.dropna(subset=['offensive']) # Remove n.a. values in columns 'Label' => check column
    return data

# Note: no need to Scale the features, Encode features, Perform cyclical engineering

def cleaning_text(sentence: str) -> str:

    # Basic cleaning
    sentence = sentence.strip() ## remove whitespaces
    sentence = sentence.lower() ## lowercase
    sentence = ''.join(char for char in sentence if not char.isdigit()) ## remove numbers



    # Advanced cleaning
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '') ## remove punctuation


    tokenized_sentence = word_tokenize(sentence) ## tokenize
    stop_words = set(stopwords.words('english')) ## define stopwords

    tokenized_sentence_cleaned = [ ## remove stopwords
        w for w in tokenized_sentence if not w in stop_words
    ]

    #remove words
    removed = ["user", "rt"]
    tokenized_sentence_cleaned = [ ## remove stopwords
        w for w in tokenized_sentence if not w in removed
    ]

    lemmatized = [
        WordNetLemmatizer().lemmatize(word, pos = "v")
        for word in tokenized_sentence_cleaned
    ]

    cleaned_sentence = ' '.join(word for word in lemmatized)

    return cleaned_sentence
