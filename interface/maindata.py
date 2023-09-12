## va organiser l'orchestration de pre donnée

## imports
import pandas as pd
import os

from moder_ia.functions import cleaning_table, cleaning_text


## preprocessing_data

raw_data = pd.read_csv(os.environ["RAW_PATH"]+"clean_dataset_v2.csv")

data_processed = cleaning_table(raw_data) # remove duplicates, ...
data_processed["text_processed"] = data_processed["text"].apply(cleaning_text) # advanced cleaning

data_processed.to_csv(os.environ["PROCESSED_PATH"]+"processed_dataset_v1.csv", index= False) # generate a file
