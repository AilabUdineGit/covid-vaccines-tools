from datetime import datetime
import pandas as pd
import tweepy
import os

from utils import pprint

def DOWNLOAD_TWEETS(DATAFRAME_SAVE_PATH, VAX, api, TWEETS_PER_DRUG, LANG=["en"]):
    # skipping VAX if the dataframe was already created
    if os.path.exists(f"{DATAFRAME_SAVE_PATH}/{VAX}_df.pkl"):
        pprint("skipping:", DATAFRAME_SAVE_PATH, "already exists")
        return
    
    TWEET_LIST = []
    for LANGUAGE in LANG:
        pprint("\n", LANGUAGE)
        
        count = 100
        query_args = build_query(VAX, count=count, lang=LANGUAGE)
        cursor = tweepy.Cursor(api.search, **query_args)
        tweet_list = []
        last_len = -1
        new_len = 0
        while len(tweet_list) < TWEETS_PER_DRUG and last_len != new_len:
            last_len = new_len
            for status in cursor.items(count):
                tweet_list.append(status.__dict__)
            new_len = len(tweet_list)
            pprint(len(tweet_list))
            
        TWEET_LIST += tweet_list
        
    pprint("total tweets:", len(TWEET_LIST))
    df = pd.DataFrame(TWEET_LIST)
    df.to_pickle(f"{DATAFRAME_SAVE_PATH}/{VAX}_df.pkl")
    pprint("saved dataframe for", VAX, "with", len(tweet_list), "tweets")

def DISPLAY_LIMITS(api):
    limits = api.rate_limit_status()
    pprint("CONTEXT:", limits["rate_limit_context"])
    for k1,v1 in limits["resources"].items():
        for k,v in v1.items():
            if "limit" not in v:
                pprint(k,v)
            if v["limit"] != v["remaining"]:
                pprint(k1,k,v)
                pprint("RESET TIME:",datetime.fromtimestamp(v["reset"]).strftime('%Y-%m-%d %H:%M:%S'))

def build_query(vaccine, no_retweets=True, lang="en", count=10, result_type="recent", until=None):
    """
----------------+----------------------------------------------------------
    no_retweets | exclude retweets? defaults to True
    lang        | defaults to "en", pass None to query all languages
    count       | defaults to 10 (number of tweets per query)
    result_type | one of "recent", "popular" or "mixed",
                | defaults to "recent"
    until       | string formatted as "YYYY-MM-DD", if None recovers
                | tweets as recent as possible
----------------+----------------------------------------------------------
    returns     | a dictionary for instanciating a tweepy.Cursor
                | on the search api, containing the keys:
                | q, tweet_mode, count, result_type, lang, until
----------------+----------------------------------------------------------
    """
    q = "covid vaccine "+vaccine.lower()
    if no_retweets:
        q = q + " -filter:retweets"
    return_dict = {
        "q": q,
        "tweet_mode": "extended",
        "count": count,
        "result_type": result_type,
    }
    if lang is not None:
        return_dict["lang"] = lang.lower()
    if until is not None:
        return_dict["until"] = until
    return return_dict
