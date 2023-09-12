from moder_ia.data_cleaning import cleaning_data
import pandas as pd

data_1 = pd.read_csv('raw_data/230911_Hate_Speech_Classification_01.csv', encoding='latin-1')
data_2 = pd.read_csv('raw_data/230911_Hate_Speech_and_Offensive_Language_01.csv')
hatespeech = pd.read_csv('raw_data/230911_Dynamically_Generated_Hate_Speech_01.csv')
hateXplain = pd.read_json('raw_data/230911_HateXplain.json')

parent=True #si notebook false
#Download
if parent==True:
    target_path='data/clean_dataset_v3.csv'
else:
    target_path='../data/clean_dataset_v3.csv'

cleaning_data(data_1,data_2,hatespeech,hateXplain).to_csv(target_path, index=False)
