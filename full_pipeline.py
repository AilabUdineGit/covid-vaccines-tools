#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import LIST_OF_VAX, pprint, NOW
from utils_01 import DOWNLOAD_TWEETS, DISPLAY_LIMITS
from utils_02 import POSTPROCESS_TWEETS
from utils_03 import INSERT_TWEETS_IN_DF
from utils_04 import PREDICT_ENTITIES
from utils_05 import INSERT_ENTITIES_IN_DF
from utils_06 import ADD_ITALIAN_TRANSLATION
from utils_07 import NORMALIZE_LOCATIONS

from CREDENTIALS import CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET

from services.tokenization_service import TokenizationService
from services.model_service import ModelService

from datetime import datetime

import pandas as pd
import tweepy
import json
import os


# In[2]:


# CONSTANTS

DATAFRAME_SAVE_PATH = "tmp_dataframes"
if not os.path.exists(DATAFRAME_SAVE_PATH):
    os.makedirs(DATAFRAME_SAVE_PATH)

TWEETS_PER_CALL = 80 #17_000
TWEETS_PER_DRUG = TWEETS_PER_CALL // len(LIST_OF_VAX)

APP = False

MAX_SEQ_LEN = 256
TS = TokenizationService()
MS = ModelService("cpu")
TOKENIZER = MS.get_tokenizer()


# In[3]:


if APP:
    auth = tweepy.AppAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
else:
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


# In[4]:


def DELETE_TEMPORARY_FILES():
    for VAX in LIST_OF_VAX:
        os.remove(f"{DATAFRAME_SAVE_PATH}/{VAX}_df.pkl")
        os.remove(f"{DATAFRAME_SAVE_PATH}/{VAX}_processed_df.pkl")
        os.remove(f"{DATAFRAME_SAVE_PATH}/{VAX}_documents.pkl")


# In[5]:


pprint(NOW.strftime('%Y-%m-%d %H:%M:%S'))

pprint()

pprint("TWEETS_PER_CALL  |", TWEETS_PER_CALL)
pprint("TWEETS_PER_DRUG |", TWEETS_PER_DRUG)

pprint()


# In[6]:


for VAX in LIST_OF_VAX:
    
    pprint("====================================================")
    pprint("\t", VAX.upper())
    
    pprint("==========[01]======================================")
    # download_tweets
    DOWNLOAD_TWEETS(DATAFRAME_SAVE_PATH, VAX, api, TWEETS_PER_DRUG)
    DISPLAY_LIMITS(api)
    
    pprint("==========[02]======================================")
    # postprocess_tweets
    POSTPROCESS_TWEETS(VAX, LIST_OF_VAX)
    
    pprint("==========[03]======================================")
    # insert_tweets_in_db
    INSERT_TWEETS_IN_DF(VAX)
    
    pprint("==========[04]======================================")
    # predict_entities
    PREDICT_ENTITIES(VAX, TS, TOKENIZER, MAX_SEQ_LEN, MS)
    
    pprint("==========[05]======================================")
    # save_entities_in_db
    INSERT_ENTITIES_IN_DF(VAX)
    
    pprint("====================================================")
    
    pprint()
    pprint()
    
pprint("==========[06]======================================")
ADD_ITALIAN_TRANSLATION()

pprint("==========[07]======================================")
NORMALIZE_LOCATIONS()

pprint("Deleting all temporary files in", DATAFRAME_SAVE_PATH)
DELETE_TEMPORARY_FILES()

THEN = datetime.now()
pprint(THEN.strftime('%Y-%m-%d %H:%M:%S'))

pprint()
pprint()

pprint("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

pprint()
pprint()

