import pandas as pd
from langdetect import detect
import tensorflow as tf
from sklearn.metrics import classification_report
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def classif_cleaning(data):

    #deleter all columns related to the annotator (i.e the person that analyzed the text)
    word='annotator'
    columns_to_drop = [col for col in data.columns if word in col]
    data.drop(columns=columns_to_drop, inplace=True)

    #exclude columns that are not used in our model
    columns_to_exclude=['comment_id','platform','sentiment', 'respect', 'insult', 'humiliate', 'status', 'dehumanize',
       'violence', 'genocide', 'attack_defend', 'hatespeech','infitms', 'outfitms', 'std_err',
       'hypothesis']
    selected_columns=[col for col in data.columns if col not in columns_to_exclude]
    data_1=data[selected_columns]

    # In order to create a Misogyny column, we needed to keep all other gender_insult and so we decide to stock it in target_gender_without_women
    columns_to_sum = ['target_gender_men',
       'target_gender_non_binary','target_gender_transgender_men',
       'target_gender_transgender_unspecified',
       'target_gender_transgender_women',
       'target_gender_other']
    data_1['target_gender_without_women'] = data_1[columns_to_sum].sum(axis=1)

    # only keep column of interest
    columns_of_interest = ['text','hate_speech_score','target_race', 'target_religion', 'target_origin', 'target_gender_women','target_gender_without_women','target_sexuality','target_age','target_disability']
    data_1=data_1[columns_of_interest]

    # cleaning of target gender without women
    data_1['target_gender_without_women']=data_1['target_gender_without_women'].replace(0,False)
    data_1['target_gender_without_women']=data_1['target_gender_without_women'].replace(1,True)
    data_1['target_gender_without_women']=data_1['target_gender_without_women'].replace(2,True)
    data_1['target_gender_without_women']=data_1['target_gender_without_women'].replace(3,True)
    data_1['target_gender_without_women']=data_1['target_gender_without_women'].replace(4,True)
    data_1['target_gender_without_women']=data_1['target_gender_without_women'].replace(5,True)
    data_1['target_gender_without_women']=data_1['target_gender_without_women'].replace(6,True)
    data_1['target_gender_without_women'].unique()

    data_1['target_gender_without_women'] = data_1['target_gender_without_women'].apply(lambda x: x.replace("0", "False") if isinstance(x, str) else x)
    data_1['target_gender_without_women'] = data_1['target_gender_without_women'].apply(lambda x: x.replace("1", "True") if isinstance(x, str) else x)

    # focus on column of interest
    data_1.drop(columns ="hate_speech_score", inplace = True)
    data_1.rename(columns={'target_race': 'racism',
              'target_religion': 'religion',
              'target_origin': 'xenophobia',
              'target_gender_women':'misogyny',
              'target_gender_without_women':'transphobia',
              'target_sexuality': 'homophobia',
              'target_age': 'ageism',
              'target_disability':'validism'}, inplace = True )

    # only keep tweets in English
    def is_english(text):
        try:
            return detect(text) == 'en'
        except:
            return False

    print("✅ english filtering done \n")

    data_1= data_1[data_1['text'].apply(is_english)]
    data_1.drop(columns ="ageism", inplace = True)
    data_1.drop(columns ="validism", inplace = True)


    #encoding categories with 0 and 1
    for row in ['racism', 'religion', 'xenophobia', 'misogyny', 'transphobia', 'homophobia']:
        data_1[row] = data_1[row].replace({False: 0, True: 1})

    data_1 = data_1.drop_duplicates() # Remove duplicates
    data_1 = data_1.dropna(subset=['text']) # Remove n.a. values in columns 'Label' => check column

    print("✅ columns cleaning done \n")

    def cleaning_table(data):
        data.drop(columns ="hate_speech_score", inplace = True)
        data.rename(columns={'target_race': 'racism',
              'target_religion': 'religion',
              'target_origin': 'xenophobia',
              'target_gender_women':'misogyny',
              'target_gender_without_women':'transphobia',
              'target_sexuality': 'homophobia',
              'target_age': 'ageism',
              'target_disability':'validism'}, inplace = True )
        data.drop(columns ="ageism", inplace = True) #removing labels ageism and validism
        data.drop(columns ="validism", inplace = True)
        return data

    def cleaning_text(sentence: str) -> str:
        # Basic cleaning
        sentence = sentence.strip() ## remove whitespaces
        sentence = sentence.lower() ## lowercase
        sentence = ''.join(char for char in sentence if not char.isdigit()) ## remove numbers

        # Advanced cleaning
        for punctuation in string.punctuation:
            sentence = sentence.replace(punctuation, '') ## remove punctuation
        tokenized_sentence = word_tokenize(sentence) ## tokenize
        stop_words = set(stopwords.words('english')) ## define stopwords

        tokenized_sentence_cleaned = [ ## remove stopwords
            w for w in tokenized_sentence if not w in stop_words
        ]

        #remove words
        removed = ["user", "rt"]
        tokenized_sentence_cleaned = [ ## remove stopwords
            w for w in tokenized_sentence if not w in removed
        ]

        lemmatized = [
            WordNetLemmatizer().lemmatize(word, pos = "v")
            for word in tokenized_sentence_cleaned
        ]

        cleaned_sentence = ' '.join(word for word in lemmatized)

        return cleaned_sentence


    data_processed = cleaning_table(data_1)
    print("✅ tables cleaned \n")

    data_processed["text_processed"] = data_processed["text"].apply(cleaning_text)
    print("✅ data processed \n")

    return data_processed
