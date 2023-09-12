import pandas as pd
import numpy as np
import os

#import all the data_set


#clearning data

def cleaning_data(data_1, data_2, hatespeech, hateXplain):

    #data_1
    new_column_data = 'Hate_Speech_Classification_01'
    new_column_name = 'source'
    data_1.insert(0, new_column_name, new_column_data)
    selected_features=['source', 'comment','label']
    data_1 = data_1[selected_features].copy()
    data_1 = data_1.rename(columns={'comment': 'text'})
    data_1 = data_1.rename(columns={'label': 'offensive'})


    #data_2
    selected_features=['class','tweet']
    data_2 = data_2[selected_features].copy()
    new_column_data = 'Hate_Speech_and_Offensive_Language'
    new_column_name = 'source'
    data_2.insert(0, new_column_name, new_column_data)
    data_2 = data_2.rename(columns={'class': 'offensive'})
    data_2 = data_2.rename(columns={'tweet': 'text'})
    data_2 = data_2[['source', 'text', 'offensive']]
    data_2['offensive'] = data_2['offensive'].replace({0: 1, 1: 1, 2:0})


    #data_3
    hatespeech["source"]="230911_Dynamically_Generated_Hate_Speech_01" # add source of the doc
    hatespeech = hatespeech[["source", "text", "label"]].copy() #select columns
    hatespeech.rename(columns={'label':'offensive'}, inplace=True) # rename column
    #encore offensive column
    mapping = {'hate': 1, 'nothate': '0'}

    hatespeech["offensive"] = hatespeech["offensive"].map(mapping)

    hatespeech

    #data_4
    hateXplain_inversed = hateXplain.transpose()
    hateXplain=hateXplain_inversed
    hateXplain["source"]="230911_HateXplain" # add source of the doc
    hateXplain = hateXplain[["post_tokens", "annotators", "source"]].copy()
    hateXplain.reset_index(inplace=True, drop=True)
    hateXplain["offensive"]=0
    hateXplain["offensive"]=hateXplain["annotators"].apply(lambda x:x[0]["label"]).apply(lambda x:0 if x =="normal" else 1)
    hateXplain = hateXplain.rename(columns={'post_tokens': 'text'})
    hateXplain = hateXplain[['source', 'text', 'offensive']].copy()
    hateXplain['text']=hateXplain['text'].astype(str)
    hateXplain['text']=hateXplain['text'].str.replace("'", "").str.replace(",", "").str.strip()



    #Concatenate_data
    concatenated_df_1= pd.concat([data_1, data_2])
    concatenated_df_1['offensive'] = concatenated_df_1['offensive'].replace({'O':0})
    concatenated_df_1['offensive']=concatenated_df_1['offensive'].astype(int)
    concatenated_df_2= pd.concat([concatenated_df_1, hatespeech])
    concatenated_df_2['offensive']=concatenated_df_2['offensive'].astype(int)
    concatenated_df_3=concatenated_df_2.reset_index(drop=True)
    concatenated_df_3=concatenated_df_2[~concatenated_df_2[['text','offensive']].duplicated()]
    concatenated_df_3=concatenated_df_3.reset_index(drop=True)
    concatenated_df_4= pd.concat([concatenated_df_3, hateXplain])
    concatenated_df_4=concatenated_df_4[~concatenated_df_4[['text','offensive']].duplicated()]
    concatenated_df_4=concatenated_df_4.reset_index(drop=True)



    return concatenated_df_4
