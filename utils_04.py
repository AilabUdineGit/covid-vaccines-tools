from services.task_loader import BertTaskLoader
from model.tester import Tester
from domain.document import Document
import pandas as pd
import pickle
import os

from utils import pprint

def PREDICT_ENTITIES(VAX, TS, TOKENIZER, MAX_SEQ_LEN, MS):
    PROCESSED_PATH = f"tmp_dataframes/{VAX}_processed_df.pkl"
    DOCUMENTS_PATH = f"tmp_dataframes/{VAX}_documents.pkl"
    
    if os.path.exists(DOCUMENTS_PATH):
        pprint("skipping:", DOCUMENTS_PATH, "already exists")
        return
    
    df = pd.read_pickle(PROCESSED_PATH)
    
    docs = df.full_text.tolist()
    tok_list = TS.tokenize_and_clean(docs)
    
    documents = [Document(doc_id, text, tokens) for doc_id, text, tokens in zip(df.tweet_id, docs, tok_list)]
    task = BertTaskLoader(TOKENIZER, documents, MAX_SEQ_LEN, annotation_type="IOB")
    
    TESTER = Tester(task, MS)
    results_df = TESTER.get_results()
    
    for document in documents:
        add_predictions_to_tokens(document, results_df)
        store_predicted_entities(document)
        
    with open(DOCUMENTS_PATH, "wb") as f:
        pickle.dump(documents, f)
    pprint("saved", DOCUMENTS_PATH)

def add_predictions_to_tokens(document, result_df):
    row = result_df.loc[document.id]
    for subtok, df_subtok, df_pred in zip(document.subtokens, row.sent, row.preds_1):
        assert subtok.text == df_subtok
        subtok.predicted_label = df_pred
    for token in document.tokens:
        if len(token.subtokens) > 0:
            subtok_labels = [st.predicted_label for st in token.subtokens]
            token.predicted_label = max(subtok_labels)

def store_predicted_entities(document):
    
    document.entities = []
    current_ent = None
    ent_start = None
    ent_end = None
    
    for tok in document.tokens:
        
        if tok.predicted_label is not None:
            
            if tok.predicted_label == 1:
                if current_ent is None:
                    current_ent = []
                current_ent.append(tok.text)
                if ent_start is None:
                    ent_start = tok.begin
                ent_end = tok.end
            
            elif tok.predicted_label == 2:
                if current_ent is not None:
                    document.entities.append((" ".join(current_ent), ent_start, ent_end))
                    ent_start = None
                    ent_end = None
                    current_ent = None
                    current_ent = []
                else:
                    current_ent = []
                current_ent.append(tok.text)
                if ent_start is None:
                    ent_start = tok.begin
                ent_end = tok.end
                    
            elif tok.predicted_label == 0:
                if current_ent is not None:
                    document.entities.append((" ".join(current_ent), ent_start, ent_end))
                    ent_start = None
                    ent_end = None
                    current_ent = None
                    
        else:
            if current_ent is not None:
                document.entities.append((" ".join(current_ent), ent_start, ent_end))
                ent_start = None
                ent_end = None
                current_ent = None