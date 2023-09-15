import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertConfig, AutoTokenizer, TFBertModel, BertTokenizer, TFBertForSequenceClassification, BertModel
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from tensorflow.keras.callbacks import EarlyStopping
from utils.bert_binary import bert_model_1

data_processed_1=pd.read_csv('data/processed_dataset_v1.csv')

bert_model_1()
