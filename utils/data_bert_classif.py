import pandas as pd
import tensorflow as tf
from transformers import BertConfig, AutoTokenizer, TFBertModel, BertTokenizer, TFBertForSequenceClassification, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import hamming_loss


def bert_classif():
    data_extracted = pd.read_csv("data/"+"labelling_dataset_v1.csv")
    texts = data_extracted['text_processed']
    labels = data_extracted.drop(['text','text_processed'],axis=1)

    #transfor the X into a list of string
    text_to_encode=texts.values.tolist()

    # Split test set
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(text_to_encode, labels, test_size=0.3, random_state=42)

    # Encode text data into BERT-friendly format

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_inputs_tokenized=tokenizer(train_inputs, add_special_tokens=True, truncation=True, max_length=200, padding='max_length', return_attention_mask=True, return_tensors='tf')

    test_inputs_tokenized=tokenizer(test_inputs, add_special_tokens=True, truncation=True, max_length=200, padding='max_length', return_attention_mask=True, return_tensors='tf')

    train_inputs_tuple = (train_inputs_tokenized['input_ids'], train_inputs_tokenized['token_type_ids'], train_inputs_tokenized['attention_mask'])
    test_inputs_tuple = (test_inputs_tokenized['input_ids'], test_inputs_tokenized['token_type_ids'], test_inputs_tokenized['attention_mask'])

    unique_classes = set()

    for label in labels:
        unique_classes.update(label)

    num_classes = len(unique_classes)

    # Step 2: Model Building
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=6)

    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.BinaryCrossentropy()


    # Compile the model
    model.compile(optimizer=optimizer, loss=loss)

    # fitting
    history = model.fit(
    train_inputs_tuple,train_labels,
    epochs=20,
    batch_size=32)


    train_predictions = model.predict(train_inputs_tuple)
    test_predictions = model.predict(test_inputs_tuple)

    # Calculate Hamming Loss for train and test separately
    threshold = 0.5
    train_hamming_loss = hamming_loss(train_labels, (train_predictions > threshold))
    test_hamming_loss = hamming_loss(test_labels, (test_predictions > threshold))

    print("Train Hamming Loss:", train_hamming_loss)
    print("Test Hamming Loss:", test_hamming_loss)

    # # Step 5: Inference
    # # You can use the trained model for inference on new text data
    # input = input("Rentrez un nouveau tweet")
    # new_texts=pd.DataFrame[input]

    # new_texts_tokenized=tokenizer(new_texts, add_special_tokens=True, truncation=True, max_length=200, padding='max_length', return_attention_mask=True, return_tensors='tf')

    # new_prediction=model.predict(new_texts_tokenized)

    return model
