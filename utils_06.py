import requests
import pandas as pd
from time import sleep
from random import randint
from tqdm.auto import tqdm
from urllib.parse import quote

from utils import OUT_DF_PATH, OUT_CSV_PATH, OUT_JSON_PATH, ENT_TRANS_PATH, update_OUT_DF
import os

def ADD_ITALIAN_TRANSLATION():
    ent_trans_df = pd.read_pickle(ENT_TRANS_PATH)

    df = ent_trans_df

    print(len(df), "entities with no translation before")

    if len(df) == 0: exit()

    def my_quote(a):
        return quote(a, safe='~@#$&()*!+=:;,.?/\'')

    def my_request_string(text, source_lang="en", target_lang="it"):
        query = my_quote(text)
        return f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={source_lang}&tl={target_lang}&dt=t&q={query}"

    def get_translation(eng_text):
        req_string = my_request_string(eng_text)
        res = requests.get(req_string)
        result = res.json()[0][0]
        it = result[0]
        en = result[1]
        if en != eng_text:
            print("original", eng_text)
            print("received", en)
            print("----------------------")
        return it

    sleep_times = [45, 34, 30, 36, 70, 40]
    # sleep_times = [5,10,30,20]
    cnt = 1
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if pd.isna(row.full_ita):
            if cnt % 1 == 0:
                df.to_pickle(ENT_TRANS_PATH)
                df.to_csv(ENT_TRANS_PATH.split(".")[0]+".csv", index=False)
            # if cnt == 4:
            #     break
            eng_text = row.full_eng
            translation = get_translation(eng_text)
            df.loc[idx,"full_ita"] = translation
            cnt+=1
            sleep(sleep_times[randint(0,len(sleep_times)-1)])
    df.to_pickle(ENT_TRANS_PATH)
    df.to_csv(ENT_TRANS_PATH.split(".")[0]+".csv", index=False)

    ent_trans_df.index = ent_trans_df.full_eng
    OUT_DF = pd.read_pickle(OUT_DF_PATH)
    for idx, row in OUT_DF.iterrows():

        for ent in row.entities:
            if ent["text"]["full_ita"] is None:
                ent["text"]["full_ita"] = ent_trans_df.loc[ent["text"]["full_eng"]].full_ita

        OUT_DF.at[idx, "entities"] = row.entities

    update_OUT_DF(OUT_DF)