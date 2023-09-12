## va appeler tout

# from interface.maindata.py import #function to prepare

import pandas as pd
from moder_ia.functions import cleaning_table, cleaning_text
from moder_ia.ml import baseline_model

def train_baseline_model(processed=True):

    ## preprocessing_data
    raw_data = pd.read_csv("data/"+"clean_dataset_v2.csv")

    data_processed = cleaning_table(raw_data) # remove duplicates, ...
    data_processed["text_processed"] = data_processed["text"].apply(cleaning_text) # advanced cleaning
    data_processed.drop(index = 82446 , inplace=True) # removing a text with "." only

    data_processed.to_csv("data/"+"processed_dataset_v1.csv", index= False) # generate a file

    if processed:
        model = baseline_model()

    else:
        model = baseline_model(processed=False)

    return model
