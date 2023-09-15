
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
from sklearn.svm import LinearSVC


import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, hamming_loss



# Machine Learning models for multi label classification

def LSTM_model(processed=True):
    '''
    '''
    data_processed = pd.read_csv("data/"+"processed_dataset_v1.csv")
    X = xx
    y = xx

    pipeline_linear_svc = make_pipeline(TfidfVectorizer(),OneVsRestClassifier(LinearSVC(), n_jobs=-1))

    if processed:
        X=

    else:
        X=

    cv_results = cross_validate(pipeline_linear_svc, X, y, cv = 5,
                                    scoring = ["accuracy","hamming_loss"],
                                    error_score='raise')



    from skmultilearn.problem_transform import LabelPowerset

powerSetSVC = LabelPowerset(LinearSVC())
powerSetSVC.fit(vectorised_train_documents, train_labels)

powerSetSVCPreds = powerSetSVC.predict(vectorised_test_documents)
metricsReport("Power Set SVC", test_labels, powerSetSVCPreds)

    return f'precision score is {average_precision}, recall score is {average_recall}, accuracy score is {average_accuracy}'
