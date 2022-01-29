import pickle
import pandas as pd
import os
from utils import OUT_DF_PATH, OUT_CSV_PATH, OUT_JSON_PATH, ENT_TRANS_PATH, update_OUT_DF

import re

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
NORM_MODALITY = "lemma"
normalizer = WordNetLemmatizer() if NORM_MODALITY == "lemma" else SnowballStemmer("english")

from utils import pprint

def INSERT_ENTITIES_IN_DF(VAX):
    DOCUMENTS_PATH = f"tmp_dataframes/{VAX}_documents.pkl"
    if not os.path.exists(DOCUMENTS_PATH):
        pprint(DOCUMENTS_PATH, "does not exist")
        return
    
    with open(DOCUMENTS_PATH, "rb") as f:
        documents = pickle.load(f)
        
    ent_trans_df = pd.read_pickle(ENT_TRANS_PATH)
    
    predictions_to_add = create_list_of_predictions(documents, ent_trans_df)
    pprint("new predictions added", len(predictions_to_add))
    
    OUT_DF = pd.read_pickle(OUT_DF_PATH)
    
    
    for tweet_id, ents in predictions_to_add.items():
        idx = OUT_DF[OUT_DF.tweet_id == tweet_id].index.tolist()[0]
        OUT_DF.at[idx, "entities"] = ents
        
        for ent in ents:
            if ent["text"]["full_eng"] not in ent_trans_df.full_eng.unique():
                ent_trans_df.loc[len(ent_trans_df)] = [
                    ent["text"]["full_eng"],
                    ent["text"]["norm_eng"],
                    ent["text"]["full_ita"]
                ]
        
    update_OUT_DF(OUT_DF)
    ent_trans_df.to_pickle(ENT_TRANS_PATH)
    ent_trans_df.to_csv(ENT_TRANS_PATH+".csv", index=False)
    

def normalize_text(text):
    original_word = text.lower()
    original_word = re.sub(" n[^\s\w]t", "n't", original_word)
    original_word = re.sub("http[^\s]+", "", original_word).strip()
    word = re.sub("[^\s\w]", " ", original_word)
    if NORM_MODALITY == "lemma":
        word = " ".join([normalizer.lemmatize(word) for word in nltk.word_tokenize(word.lower())])
    else:
        word = " ".join([normalizer.stem(word) for word in nltk.word_tokenize(word.lower())])
        
    clean_key_parts = [k for k in nltk.word_tokenize(word) if k not in STOPWORDS]
    word = " ".join(clean_key_parts)
    return word, original_word


def create_list_of_entities(doc, ent_trans_df):
    entities_list = []
    if len(doc.entities) > 0:
        for ent in doc.entities:
            word, original_word = normalize_text(ent[0])
            
            prev_value = ent_trans_df[ent_trans_df.full_eng == ent[0]]
            if len(prev_value)>0 and prev_value.full_ita.iloc[0] is not None:
                full_ita = prev_value.full_ita.iloc[0]
            else:
                full_ita = None
            
            entity = dict(
                position = dict(start=ent[1], end=ent[2]),
                text = dict(full_eng=ent[0], norm_eng=word, full_ita=full_ita)
            )
            entities_list.append(entity)
            
    return entities_list
            

def create_list_of_predictions(documents, ent_trans_df):
    predictions = dict()
    for doc in documents:
        pred = create_list_of_entities(doc, ent_trans_df)
        predictions[doc.id] = pred
    return predictions