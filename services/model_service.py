from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertConfig
from transformers import AutoTokenizer, AutoConfig
from transformers import AdamW
import torch
import os
from model.bert_crf import Bert_CRF
import json


class ModelService(object):
    
    def __init__(self, device):
        self.MODEL_PATH = "model/ML4H_blind_smm4h_BERT_CRF"
        self.config = AutoConfig.from_pretrained(self.MODEL_PATH)
        self.tokenizer = None
        self.model = None
        self.MODEL_CLASS = Bert_CRF
        self.device = device
        
    def get_config(self):
        return self.config
        
    def get_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, config=self.config)
        return self.tokenizer

    def get_model(self):
        if self.model is None:
            self.model = self.MODEL_CLASS.from_pretrained(self.MODEL_PATH, config=self.config)
            self.model.to(self.device) # Runs the model on the GPU (if available)
        return self.model