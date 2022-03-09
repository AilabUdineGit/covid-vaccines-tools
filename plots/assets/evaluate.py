#!/usr/bin/env python
# coding: utf-8

import pandas as pd

from ner_eval import Evaluator
from ast import literal_eval
import json

from spacy.lang.en import English
from spacy.gold import biluo_tags_from_offsets as get_biluo

from sklearn.metrics import f1_score, precision_score, recall_score
AVG_MODE = "binary"

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter, 
    description = """

Expected input: a tab separated file (with header)
containing at least the "General" fields and the ones
needed to evaluate the predictions for one or more of
the following tasks: "Entity Extraction", "Binary Classification".

Columns

  * General
    - doc_id: (optional) unique identifier for the document
    - text: full text of the document on which predictions are made

  * Entity Extraction
    - gold_str: list of real entities in the text as strings (optional, for sanity check)
    - pred_str: list of predicted entities as strings (optional, for sanity check)
    - gold_int: list of real entities in the text, given as character indexes
                (start included, end excluded) + entity type (string)
    - pred_int: list of predicted entities, given as character indexes
                (start included, end excluded) + entity type (string)
    
      gold_int and pred_int must be serializations of valid python lists
      examples: [(0,5,"AE"),(10,15,"Drug")]
                [[0,5,"AE"],[10,15,"Drug"]]

  * Binary Classification
    - gold_class: int, actual class label
    - pred_class: int, predicted class label
    
""")
parser.add_argument('-N', '--ner', action="store_true", help="Flag to evaluate entity extraction (NER)")
parser.add_argument('-B', '--binary', action="store_true", help="Flag evaluate binary classification")
parser.add_argument('-P', '--path', type=str, help="Path of the file containing the predictions")
parser.add_argument('--sep', type=str, default="comma")
args = parser.parse_args()
PATH = args.path
BIN = args.binary
NER = args.ner
SEP = "\t" if args.sep=="tab" else ","

def fix_ent(doc, ent):
        
    start,end,ent_type = ent
    
    mod = False
    
    for tok in doc:
        tstart = tok.idx
        tend = tok.idx + len(tok)
        
        if tstart < start < tend:
            start = tstart
            break
        elif start == tstart-1:
            start += 1
            mod = True
            break
            
    for tok in doc:
        tstart = tok.idx
        tend = tok.idx + len(tok)
        
        if tstart < end < tend:
            end = tend
            break
        elif end == tend+1:
            end -= 1
            mod = True
            break
            
#     if mod:
#         print(doc)
#         print(start, end)
#         print("|"+doc.text[start:end]+"|")
#         print("-"*10)
        
    return (start, end, ent_type)

def report(txt):
#     with open(report_path, "a") as f:
#         f.write(txt+"\n")
    print(txt)



# doc_id, text, gold_int, gold_str, pred_int, pred_str, gold_class, pred_class
df = pd.read_csv(PATH, header=0, dtype=str, sep=SEP)

root_name = ".".join(PATH.split(".")[:-1])
report_path = root_name+"_evaluation_report.txt"
# with open(report_path, "w") as f:
#     f.write("")

if BIN:
    assert "gold_class" in df.columns
    assert "pred_class" in df.columns

if NER:
    assert "gold_int" in df.columns
    assert "pred_int" in df.columns
    
if NER:
    df.gold_int = df.gold_int.apply(lambda x: literal_eval(x))
    #df.gold_str = df.gold_str.apply(lambda x: literal_eval(x))
    df.pred_int = df.pred_int.apply(lambda x: literal_eval(x))
    #df.pred_str = df.pred_str.apply(lambda x: literal_eval(x))
    
if BIN:
    
    df.gold_class = df.gold_class.astype("int")
    df.pred_class = df.pred_class.astype("int")
    
    call_args = {
        "y_true": df.gold_class.tolist(),
        "y_pred": df.pred_class.tolist(),
        "average": AVG_MODE,
        "pos_label": 1,
        "zero_division": 0,
    }
    f1 = f1_score(**call_args)
    precision = precision_score(**call_args)
    recall = recall_score(**call_args)
    
    report("-----------------------")
    report("Binary Classification")
    
    for m,v in zip(["precision", "recall", "f1"], [precision, recall, f1]):
        report(f"{m:<10}| {round(v,4)}")
    

if NER:
    nlp = English()
    
    df["spacy_doc"] = df.text.apply(lambda x: nlp(x))
    
    df["gold_fix"] = df.apply(lambda x: [fix_ent(x.spacy_doc,y) for y in x.gold_int], axis=1)
    df["pred_fix"] = df.apply(lambda x: [fix_ent(x.spacy_doc,y) for y in x.pred_int], axis=1)

    df["gold_tags"] = None
    df["pred_tags"] = None

    for idx, x in df.iterrows():

        try:
            gold_tags = get_biluo(x.spacy_doc, x.gold_fix)
        except ValueError:
            print("GOLD", x)
            gold_tags = []

        try:
            pred_tags = get_biluo(x.spacy_doc, x.pred_fix)
        except ValueError:

            print("PREDS", x)
            pred_tags = []

        gold_tags = [tag.replace("U-","B-").replace("L-","I-") for tag in gold_tags]
        pred_tags = [tag.replace("U-","B-").replace("L-","I-") for tag in pred_tags]

        df.loc[idx, "gold_tags"] = gold_tags
        df.loc[idx, "pred_tags"] = pred_tags
    
#     df["gold_tags"] = df.apply(lambda x: get_biluo(x.spacy_doc, x.gold_fix), axis=1)
#     df["pred_tags"] = df.apply(lambda x: get_biluo(x.spacy_doc, x.pred_fix), axis=1)
    
#     df["gold_tags"] = df.apply(lambda x: [tag.replace("U-","B-").replace("L-","I-") for tag in x.gold_tags], axis=1)
#     df["pred_tags"] = df.apply(lambda x: [tag.replace("U-","B-").replace("L-","I-") for tag in x.pred_tags], axis=1)
    
    all_ent_types = set([y.split("-")[-1] for x in df.pred_tags.tolist() for y in x]) - set("O")
    
    evaluator = Evaluator(df.gold_tags.tolist(), df.pred_tags.tolist(), all_ent_types)
    all_metrics, type_metrics = evaluator.evaluate()
    
#     with open(root_name+"_all_ner_metrics.json", "w") as f:
#         json.dump(all_metrics, f, indent=4)
        
#     report("-----------------------")
#     report("NER Strict Metrics")
#     for m in ["precision", "recall", "f1"]:
#         report(f"{m:<10}| {all_metrics['strict'][m]}")
#     report("-----------------------")
#     report("NER Relaxed Metrics")
#     for m in ["precision", "recall", "f1"]:
#         report(f"{m:<10}| {round(all_metrics['partial'][m],4)}")
    print(round(all_metrics['partial']["f1"],4), ", # f1")
    print(round(all_metrics['partial']["precision"],4), ", # p")
    print(round(all_metrics['partial']["recall"],4), ", # r")
    print("# ----------------")








