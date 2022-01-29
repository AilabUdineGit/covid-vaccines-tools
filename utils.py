from datetime import datetime
import json
import os
import pandas as pd

TIMEDELTA = 7

with open("VACCINES_TO_TRACK", "r") as f:
    LIST_OF_VAX = [line.strip() for line in f.readlines()]
    LIST_OF_VAX = [line for line in LIST_OF_VAX if len(line)>0]

NOW = datetime.now()
LOG_FILE = f"logs/{NOW.year}_{NOW.month}_{NOW.day}.log"
if not os.path.exists("logs"):
    os.makedirs("logs")

MAIN_JSON_PATH = "info.json"
# if not os.path.exists(MAIN_JSON_PATH):
#     with open(MAIN_JSON_PATH, "w") as f:
#         json.dump(dict(), f)

    
def pprint(*text):
    real_text = " ".join([str(t) for t in text])
    print(real_text)
    with open(LOG_FILE, "a") as f:
        f.write(real_text)
        f.write("\n")

OUT_DF_PATH = "output_dataframe.pkl"
OUT_CSV_PATH = "output_dataframe.csv"
OUT_JSON_PATH = "output_dataframe.json"
ENT_TRANS_PATH = "entity_translation.pkl"

def update_OUT_DF(new_df):
    new_df.to_pickle(OUT_DF_PATH)
    new_df.to_csv(OUT_CSV_PATH, index=False)
    new_df.to_json(OUT_JSON_PATH, orient="records", index=True)

if not os.path.exists(ENT_TRANS_PATH):
    ent_trans_df = pd.DataFrame(columns=["full_eng", "norm_eng", "full_ita"])
    ent_trans_df.to_pickle(ENT_TRANS_PATH)
    
if not os.path.exists(OUT_DF_PATH):
    OUT_DF = pd.DataFrame(columns=[
        "tweet_id",
        "created_at",
        "text_for_unique",
        "lang",
        "user_location_norm",
        "country",
        "ent_hashtags",
        "ent_urls",
        "vaccines",
        "sentiment_value",
        "entities",
    ])
    update_OUT_DF(OUT_DF)
