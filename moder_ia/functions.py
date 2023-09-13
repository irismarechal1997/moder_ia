import pandas as pd
import numpy as np
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



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

# DEEP LEARNING MODEL

import numpy as np
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

### Let's create some mock data
def get_mock_up_data():

    data_processed = pd.read_csv("data/"+"processed_dataset_v1")

    X = data_processed["text"]
    y = data_processed["offensive"]

    # for i in np.range(1, len(data_processed)) :
    #     X.append([data_processed.loc[i, "text"]])
    #     y.append([1])

    ### Let's tokenize the vocabulary
    tk = Tokenizer()
    tk.fit_on_texts(X)
    vocab_size = len(tk.word_index)
    print(f'There are {vocab_size} different words in your corpus')
    X_token = tk.texts_to_sequences(X)

    ### Pad the inputs
    X_pad = pad_sequences(X_token, dtype='float32', padding='post')

    return X_pad, y, vocab_size

X_pad, y, vocab_size = get_mock_up_data()

import tensorflow_datasets as tfds
from keras_preprocessing.text import text_to_word_sequence


# def load_data(percentage_of_sentences=None):
    # data_processed = pd.read_csv("data/"+"processed_dataset_v1")

    # X = data_processed["text"]
    # y = data_processed["offensive"]

#     train_data, test_data =

#     train_sentences, y_train = tfds.as_numpy(train_data)
#     test_sentences, y_test = tfds.as_numpy(test_data)

#     # Take only a given percentage of the entire data
#     if percentage_of_sentences is not None:
#         assert(percentage_of_sentences> 0 and percentage_of_sentences<=100)

#         len_train = int(percentage_of_sentences/100*len(train_sentences))
#         train_sentences, y_train = train_sentences[:len_train], y_train[:len_train]

#         len_test = int(percentage_of_sentences/100*len(test_sentences))
#         test_sentences, y_test = test_sentences[:len_test], y_test[:len_test]

#     X_train = [text_to_word_sequence(_.decode("utf-8")) for _ in train_sentences]
#     X_test = [text_to_word_sequence(_.decode("utf-8")) for _ in test_sentences]

#     return X_train, y_train, X_test, y_test

# X_train, y_train, X_test, y_test = load_data(percentage_of_sentences=10)


from gensim.models import Word2Vec

word2vec = Word2Vec(sentences=X_train)
wv = word2vec.wv
