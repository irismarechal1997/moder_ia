from utils.data_cleaning import cleaning_data
from utils.data_preproc import cleaning_table, cleaning_text

from utils.data_classif_cleaning import classif_cleaning
import pandas as pd

def data_processed(parent=True):
    ## Concatenate and pre clean data
    data_1 = pd.read_csv('raw_data/230911_Hate_Speech_Classification_01.csv', encoding='latin-1')
    data_2 = pd.read_csv('raw_data/230911_Hate_Speech_and_Offensive_Language_01.csv')
    hatespeech = pd.read_csv('raw_data/230911_Dynamically_Generated_Hate_Speech_01.csv')
    hateXplain = pd.read_json('raw_data/230911_HateXplain.json')
    data_5=pd.read_csv('raw_data/11092023_Happy_Tweet.csv')

    if parent==True:
        target_path='data/clean_dataset_v6.csv'
    else:
        target_path='../data/clean_dataset_v6.csv'
    cleaning_data(data_1,data_2,hatespeech,hateXplain, data_5).to_csv(target_path, index=False)

    clean_data = pd.read_csv("data/clean_dataset_v6.csv") # assign variable
    print("✅ clean_data_set generated \n")

    ## Pre_process data
    data_processed = cleaning_table(clean_data) # remove duplicates, ...
    data_processed["text_processed"] = data_processed["text"].apply(cleaning_text) # advanced cleaning

    data_processed.to_csv("data/processed_dataset_v1.csv", index= False) # generate a file


    print("✅ preprocess() done, processed_dataset generated \n")
    return data_processed


def data_labels_processed():
    data=pd.read_csv('raw_data/measuring_hate_speech.csv')
    print("✅ preprocess() done,label_dataset generated \n")
    label_dataset = classif_cleaning(data)
    label_dataset.to_csv("data/labelling_dataset_v1.csv", index= False) # generate a file
    return label_dataset



if __name__ == "__main__": ## dire quelle fonction
    data_labels_processed()
