# import packages

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB



def crossval_baseline_model(processed=True):

    data_processed = pd.read_csv("data/"+"processed_dataset_v1.csv")
    X_proc = data_processed["text_processed"]
    X = data_processed["text"]
    y = data_processed["offensive"]

    pipeline_naive_bayes = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB())

    if processed:
        cv_results = cross_validate(pipeline_naive_bayes, X_proc, y, cv = 5,
                                    scoring = ["precision","recall", "accuracy"],
                                    error_score='raise')
        average_precision = cv_results["test_precision"].mean()


    else:
        cv_results = cross_validate(pipeline_naive_bayes, X, y, cv = 5,
                                    scoring = ["precision","recall", "accuracy"],
                                    error_score='raise')
        average_precision = cv_results["test_precision"].mean()

    average_precision = round(cv_results["test_precision"].mean(),2)
    average_recall = round(cv_results["test_recall"].mean(),2)
    average_accuracy = round(cv_results["git test_accuracy"].mean(),2)

    return f'precision score is {average_precision}, recall score is {average_recall}, accuracy score is {average_accuracy}'


def baseline_model(processed=True):

    data_processed = pd.read_csv("data/"+"processed_dataset_v1.csv")
    X_proc = data_processed["text_processed"]
    X = data_processed["text"]
    y = data_processed["offensive"]

    pipeline_naive_bayes = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB())

    if processed:
        model = pipeline_naive_bayes.fit(X_proc, y)

    else:
        model = pipeline_naive_bayes.fit(X, y)

    print(f"model trained, accuracy score is {round(model.score(X,y),2)}, model with {processed=}")
    return model
