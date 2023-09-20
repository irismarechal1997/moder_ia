import pandas as pd
import tensorflow as tf
from transformers import BertConfig, AutoTokenizer, TFBertModel, BertTokenizer, TFBertForSequenceClassification, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# def hamming_loss(y_true, y_pred, threshold=0.5):
#     y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
#     y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

# # Apply the threshold to the predicted values
#     y_pred = tf.cast(y_pred > threshold, dtype=tf.float32)

#  # Calculate the Hamming Loss\
#     epsilon = 1e-7  # Small epsilon value\n",
#     hamming_loss = 1.0 - tf.reduce_mean(tf.reduce_sum(y_true * y_pred, axis=1) / (tf.reduce_sum(y_true + y_pred - y_true * y_pred, axis=1) + epsilon))

#     return hamming_loss.numpy()


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

    train_inputs_tokenized=tokenizer(train_inputs, max_length=200, padding='max_length', return_attention_mask=True, return_tensors='tf')

    test_inputs_tokenized=tokenizer(test_inputs, max_length=200, padding='max_length', return_attention_mask=True, return_tensors='tf')

    train_inputs_tuple = (train_inputs_tokenized['input_ids'], train_inputs_tokenized['token_type_ids'], train_inputs_tokenized['attention_mask'])
    test_inputs_tuple = (test_inputs_tokenized['input_ids'], test_inputs_tokenized['token_type_ids'], test_inputs_tokenized['attention_mask'])

    unique_classes = set()

    for label in labels:
        unique_classes.update(label)

    num_classes = len(unique_classes)

    # Step 2: Model Building
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=6)

    for layer in model.layers[:-2]:
        layer.trainable=False

    loss = tf.keras.losses.BinaryCrossentropy()


    # Compile the model
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy', 'Recall', 'Precision' ])

    # fitting
    history = model.fit(
    train_inputs_tuple,train_labels,
    epochs=5,
    batch_size=128,
    validation_split=0.1)

    print(model.summary())

    # train_predictions = model.predict(train_inputs_tuple)
    # test_predictions = model.predict(test_inputs_tuple)

    # # Calculate Hamming Loss for train and test separately
    # train_hamming_loss = hamming_loss(train_predictions, train_labels)
    # test_hamming_loss = hamming_loss(test_predictions, test_labels)

    # print("Train Hamming Loss:", train_hamming_loss)
    # print("Test Hamming Loss:", test_hamming_loss)

    # # Step 5: Inference
    # # You can use the trained model for inference on new text data
    # input = input("Rentrez un nouveau tweet")
    # new_texts=pd.DataFrame[input]

    # new_texts_tokenized=tokenizer(new_texts, add_special_tokens=True, truncation=True, max_length=200, padding='max_length', return_attention_mask=True, return_tensors='tf')

    # new_prediction=model.predict(new_texts_tokenized)

    return history, model
