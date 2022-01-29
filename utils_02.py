import pandas as pd
import os

from utils import pprint

def POSTPROCESS_TWEETS(VAX, LIST_OF_VAX):
    ORIGINAL_DF_PATH = f"tmp_dataframes/{VAX}_df.pkl"
    PROCESSED_DF_PATH = f"tmp_dataframes/{VAX}_processed_df.pkl"
    if not os.path.exists(ORIGINAL_DF_PATH):
        pprint(ORIGINAL_DF_PATH, "does not exists")
        return
    if os.path.exists(PROCESSED_DF_PATH):
        pprint("skipping:", PROCESSED_DF_PATH, "already exists")
        return
    df_p = pd.read_pickle(ORIGINAL_DF_PATH)
    new_df = init_clean_df(df_p)
    del df_p
    new_df = add_more_information(new_df, LIST_OF_VAX)
    new_df.to_pickle(PROCESSED_DF_PATH)
    pprint("saved processed dataframe for", VAX, "with", len(new_df), "tweets")

def init_clean_df(df_p):
    new_df = pd.DataFrame(df_p._json.tolist())
    new_df.drop([
        "id_str","in_reply_to_status_id_str",
        "truncated","display_text_range","metadata",
        "favorite_count","favorited","retweeted","retweet_count",
        "in_reply_to_screen_name","in_reply_to_user_id_str","in_reply_to_user_id","in_reply_to_status_id",
        "extended_entities","is_quote_status",
        "contributors"], axis=1, inplace=True)
    if "quoted_status" in new_df.columns:
        new_df.drop(['quoted_status_id', 'quoted_status_id_str', 'quoted_status'], axis=1, inplace=True)
    new_df.rename(columns={'source':'source_url', 'id':'tweet_id'}, inplace=True)
    new_df["source"] = df_p.source
    new_df["created_at"] = df_p.created_at
    new_df["api_query"] = [ api.last_response.url for api in df_p._api.tolist()]
    new_df.possibly_sensitive.fillna(False, inplace=True)
    return new_df

def add_vaccines(row, LIST_OF_VAX):
    vaccines_in_tweet = set()
    tweet_text = row.full_text_no_retweet.lower()
    for vax_name in LIST_OF_VAX:
        if "johnson" in vax_name and "johnson" in tweet_text:
            vaccines_in_tweet.add(vax_name)
        if vax_name in tweet_text:
            vaccines_in_tweet.add(vax_name)
        elif vax_name.split()[0] in row.api_query.lower():
            vaccines_in_tweet.add(vax_name)
    vaccines_in_tweet = list(vaccines_in_tweet)
    vaccines_in_tweet.sort()
    row.vaccines = tuple(vaccines_in_tweet)
    return row

def add_user_info(row):
    row.user_id = row.user["id"]
    row.user_name = row.user["screen_name"]
    row.user_location = row.user["location"]
    return row

def add_entities(row):
    for key in ["hashtags","symbols","user_mentions","urls","media"]:
        if key in row.entities:
            row["ent_"+key] = row.entities[key]
        else:
            row["ent_"+key] = list()
    return row

def add_country_info(row):
    if not pd.isna(row.place):
        row.place_id = row.place["id"]
        row.place_name = row.place["full_name"]
        row.place_country = row.place["country"]
    else:
        row.place_id = ""
        row.place_name = ""
        row.place_country = ""
    return row

def add_retweet_information(row):
    
    if "retweeted_status" in row and not pd.isna(row.retweeted_status):
        retweet_text = row.full_text
        original_text = row["retweeted_status"]["full_text"]
        added_by_retweeter = retweet_text[3:retweet_text.find(":")]
        
        if added_by_retweeter[1:] == row.retweeted_status["user"]["screen_name"]:
            added_by_retweeter = ""
        
        row.is_retweet = True
        row.full_text_no_retweet = original_text
        row.text_added_by_retweet = added_by_retweeter
        
        if not original_text.startswith(retweet_text[retweet_text.find(":")+2:-1]):
            pprint("Mismatch found between the retweeted text\n", retweet_text[retweet_text.find(":")+2:-1],
                "\nand the original text\n", original_text, "\nMismatch will be solved by using the original text.\n")
    else:
        row.is_retweet = False
        row.full_text_no_retweet = row.full_text
        row.text_added_by_retweet = ""
    
    return row

def add_more_information(new_df, LIST_OF_VAX):
    new_df["is_retweet"] = None
    new_df["full_text_no_retweet"] = None
    new_df["text_added_by_retweet"] = None
    new_df = new_df.apply(lambda row: add_retweet_information(row), axis=1)

    new_df["place_id"] = None
    new_df["place_name"] = None
    new_df["place_country"] = None
    new_df = new_df.apply(lambda row: add_country_info(row), axis=1)


    new_df["ent_hashtags"] = None
    new_df["ent_symbols"] = None
    new_df["ent_media"] = None
    new_df["ent_user_mentions"] = None
    new_df["ent_urls"] = None
    new_df = new_df.apply(lambda row: add_entities(row), axis=1)

    new_df.drop("entities", axis=1, inplace=True)

    new_df["user_id"] = None
    new_df["user_name"] = None
    new_df["user_location"] = None
    new_df = new_df.apply(lambda row: add_user_info(row), axis=1)
    
    new_df["vaccines"] = None
    new_df = new_df.apply(lambda row: add_vaccines(row, LIST_OF_VAX), axis=1)

    return new_df
