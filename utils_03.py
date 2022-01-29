import pandas as pd
import json
import os
import re
from tqdm import tqdm

from utils import pprint
from utils import OUT_DF_PATH, OUT_CSV_PATH, OUT_JSON_PATH, ENT_TRANS_PATH, update_OUT_DF

def INSERT_TWEETS_IN_DF(VAX):
    PROCESSED_PATH = f"tmp_dataframes/{VAX}_processed_df.pkl"
    if not os.path.exists(PROCESSED_PATH):
        pprint(PROCESSED_PATH, "does not exist")
        return
    OUT_DF = pd.read_pickle(OUT_DF_PATH)
    
    pprint("tweets before:", len(OUT_DF))
    
    df = pd.read_pickle(PROCESSED_PATH)
    pprint(len(df), "preprocessed tweets")
    
    tweets_to_add = add_tweets_to_df(OUT_DF, df)
    pprint(len(tweets_to_add), "actual new tweets")
    
    sentiment = get_sentiment2([tweet["full_text"] for tweet in tweets_to_add])
    assert len(tweets_to_add) == len(sentiment)
    for i,s in enumerate(sentiment):
        tweets_to_add[i]["sentiment_value"] = s
        del tweets_to_add[i]["full_text"]
        
    tweets_to_add = pd.DataFrame(tweets_to_add, columns=list(OUT_DF.columns))
    
    OUT_DF = pd.concat((OUT_DF, tweets_to_add), ignore_index=True)
    update_OUT_DF(OUT_DF)
    
    pprint("tweets after:", len(OUT_DF))
    
    ok_ids = [tweet for tweet in tweets_to_add.tweet_id]
    df.index = df.tweet_id
    new_df = df.loc[ok_ids]
    new_df.index = range(len(new_df))
    
    pprint("overwriting df (keeping new tweets only)")
    new_df.to_pickle(PROCESSED_PATH)
    pprint("new len", len(new_df))
        


def get_superclean_text(text):
    text1 = re.sub("http[^\s]+", "", text).strip()
    text1 = re.sub("@[^\s]+", "", text1)
    text1 = re.sub("#[^\s]+", "", text1)
    text1 = re.sub("\$[^\s]+", "", text1)
    text1 = re.sub("[^\s\w]", " ", text1)
    text1 = re.sub("\s+", " ", text1).strip().lower()
    #for c in punctuation:
    #    text2 = text2.replace(c, "")
    text_pieces = text1.split()
    clean_text = " ".join(text_pieces[:10])
    return clean_text
    
from string import punctuation
import re

def normalize(text):
    for c in punctuation:
        text = text.replace(c, " ")
    text = text.encode("ascii", "ignore").decode()
    text = re.sub("\s+", " ", text)
    return text.lower().strip()
    

def add_tweets_to_df(OUT_DF, df):
    tweets = []
    for _, row in df.iterrows():
        if not row.tweet_id in OUT_DF.tweet_id.unique():
            loc = row.place_country if len(row.place_country)>0 else row.user_location
            norm_loc = normalize(loc)
            created_at_date = row.created_at.to_pydatetime()
            created_at_date = created_at_date.strftime("%Y-%m-%d")
            tweets.append(dict(
                tweet_id=row.tweet_id,
                created_at=created_at_date,
                full_text=row.full_text_no_retweet,
                text_for_unique=get_superclean_text(row.full_text_no_retweet),
                lang=row.lang,
                user_location_norm=norm_loc if len(norm_loc)>1 else "",
                country="",
                ent_hashtags=row.ent_hashtags,
                ent_urls=row.ent_urls,
                vaccines=row.vaccines,
                sentiment_value=None,
                entities=[],         #"[(entity_text_eng, entity_norm_eng, entity_text_ita)]",
            ))
    return tweets
    
    
import json
from transformers import logging
logging.set_verbosity_error()
from transformers import pipeline


PIPELINE = pipeline(task='sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment")#, device=0)

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def _sentiment_score(label, certainty):
    label = int(label.replace("LABEL_",""))
    if label == 1:
        label = 0.0001
    if label == 2:
        label = 1
    if label == 0:
        label = -1
    return label * certainty
        
def _chunk(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def get_sentiment2(sentences):
    
    sentences = [preprocess(sent) for sent in sentences]
    
    out_list = []
    
    for sents in tqdm(_chunk(sentences, 300), desc="roberta sentiment"):
        out = PIPELINE(sents)
        # out_list += [_sentiment_score(o["label"],o["score"]) for o in out]
        out_list += out

    return out_list

