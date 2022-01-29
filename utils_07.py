import requests
import pandas as pd
from time import sleep
from random import randint
from tqdm.auto import tqdm
from urllib.parse import quote

from string import punctuation
from fuzzywuzzy import fuzz
import re

import requests
import json
#from bs4 import BeautifulSoup
from ast import literal_eval

from utils import MAIN_JSON_PATH
from utils import OUT_DF_PATH, OUT_CSV_PATH, OUT_JSON_PATH, ENT_TRANS_PATH, update_OUT_DF

from io import open

def normalize(text):
    for c in punctuation:
        text = text.replace(c, " ")
    text = text.encode("ascii", "ignore").decode()
    text = re.sub("\s+", " ", text)
    text = text.lower().strip()
    return text

def NORMALIZE_LOCATIONS():

    LOC_PATH = "locations.pkl"

    OUT_DF = pd.read_pickle(OUT_DF_PATH)
    locations = OUT_DF[OUT_DF.user_location_norm.str.len()>0].user_location_norm.unique().tolist()
    print(len(locations), "tweets with location")
    print("Some locations")
    print(locations[:5])

    clean_loc = [normalize(loc) for loc in locations]
    clean_loc = [x for x in clean_loc if len(x)>1]
    print(len(set(clean_loc)), "unique cleaned locations")
    print("Some clean_loc")
    print(clean_loc[:5])

    df = pd.DataFrame(clean_loc, columns=["clean"])
    new_df = df.clean.value_counts()
    new_df.name = "counts"

    df = pd.DataFrame(new_df)
    df["clean"] = df.index
    df.index = range(len(df))
    df["google"] = None
    df["code"] = None

    import os
    if os.path.exists(LOC_PATH):
        df_old = pd.read_pickle(LOC_PATH)
        df_old_unique = set(df_old.clean.unique())
        df["new"] = df.clean.apply(lambda x: not x in df_old_unique)
        df_new = df[df.new].drop("new", axis=1)
        df = pd.concat((df_old,df_new))
        df.index = range(len(df))
    print(df)

    df.to_pickle(LOC_PATH)
    df.to_csv(LOC_PATH.split(".")[0]+".csv")
    print(df)

    def update_info(df):

        WARN_WORDS = [
            "world", "anywhere", "everywhere", "internet", "online",
            "rainbow", "earth", "glob", "where", "universe", "your",
             "twitter", "paradise", "http", "www", "round", "house", "home", "void",
            "virtual", "here", "worl", "space", "galaxy", "milky"
        ]

        DF = df[~df.code.isna()].copy()
        for idx, row in DF.iterrows():
            if row.code == "Srinagar":
                DF.loc[idx, "code"] = "IN"
            if row.code != "ERROR":
                none_count = sum([1 if x is None else 0 for x in row.google[1]])
                first_where = " ".join(x[0] for x in row.google[0][0][1]).lower()
                ratio = fuzz.token_sort_ratio(row.clean, first_where)
                if any([W in row.clean for W in WARN_WORDS]) or \
                   (none_count <= 1 and ratio < 50) or \
                    row.clean.replace(" ","").isdigit():
                    DF.loc[idx, "code"] = "WRONG"

        to_drop = DF[DF.code == "WRONG"].index.tolist() + \
                  DF[DF.code == "ERROR"].index.tolist() + \
                  DF[DF.code.isna()].index.tolist()
        DF = DF.drop(to_drop)

        tmp = DF.groupby("code").counts.sum().to_dict()
        new = [{"code": k, "count": v} for k,v in tmp.items()]
        new.sort(key = lambda x: x["count"])

        with open(MAIN_JSON_PATH, "r") as f:
            json_content = json.load(f)

        json_content["countries"] = new

        with open(MAIN_JSON_PATH, "w", encoding="ascii") as f:
            json.dump(json_content, f)


    def my_quote(a):
        return quote(a, safe='~@#$&()*!+=:;,.?/\'')

    def get_google(text):
        try:
            QUERY = my_quote(text.replace(" ", "+"))
            url = f"https://www.google.com/maps/search/{QUERY}?hl=en"
            cookies = {"CONSENT": "YES+cb.20210622-13-p0.it+FX+362"}
            res = requests.get(url,
            #headers={
            #'User-Agent':
            #'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_2) '\
            #'AppleWebKit/537.36 (KHTML, like Gecko) '\
            #'Chrome/55.0.2883.95 '\
            #'Safari/537.36'},
            cookies=cookies, timeout=20)
            # res = requests.get(url)
            html = res.content.decode()
            idx = html.find("[[[7,")
            cnt = 0
            for c, char in enumerate(html[idx:]):
                if char == "[":
                    cnt += 1
                if char == "]":
                    cnt -= 1
                if cnt == 0:
                    break
            clean = html[idx:idx+c+1]
            clean = clean.replace("\\n", "").replace('\\"', '"').replace("null", "None")
            lit = literal_eval(clean)

            if type(lit[1][-1])==str and len(lit[1][-1])==2:
                return lit, lit[1][-1]
            else:
                match = re.search("(\'|\")[A-Z][A-Z](\'|\")", clean)
                if match:
                    return lit, match.group(0).replace('"', '').replace("'", "")
                else:
                    print("ERROR_match")
                    return "ERROR", "ERROR"
        except:
            print("ERROR")
            return "ERROR", "ERROR"

    update_info(df)

    sleep_times = [45, 34, 30, 36, 40]
    # sleep_times = [5,10,30,20]
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if row.google is None or row.code is None:#pd.isna(row.google):
            text = row.clean
            google, code = get_google(text)

            if code != "ERROR":
                OUT_DF.country.where(OUT_DF.user_location_norm!=text, code, inplace=True)
                update_OUT_DF(OUT_DF)

            df.at[idx,"google"] = google
            df.at[idx,"code"] = code
            df.to_pickle(LOC_PATH)
            df.to_csv(LOC_PATH.split(".")[0]+".csv")
            if idx % 10 == 0:
                update_info(df)
            sleep(sleep_times[randint(0,len(sleep_times)-1)])
        
    df = df[df.code != "ERROR"]
    df.index = range(len(df))
    df.to_pickle(LOC_PATH)
    df.to_csv(LOC_PATH.split(".")[0]+".csv")
    update_info(df)
    print("Errors:", df[df.google == "ERROR"].shape[0])
    print("None:", df.google.isna().sum())
    print("Ok:", len(df[df.google != "ERROR"]) - df.google.isna().sum())
    
    for idx, row in df.iterrows():
        code = row.code
        text = row.clean
        if code is not None and code != "ERROR" and code != "WRONG":
            OUT_DF.country.where(OUT_DF.user_location_norm!=text, code, inplace=True)
    update_OUT_DF(OUT_DF)