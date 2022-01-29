from transformers import BertPreTrainedModel, BertForTokenClassification, AutoModel
import torch.nn.functional as F
from torchcrf import CRF
from torch import nn
import torch
log_soft = F.log_softmax

#import logging
#logger = logging.getLogger(__name__)


class Bert_CRF(BertPreTrainedModel):
    # idea:
    # see also: https://github.com/Dhanachandra/bert_crf/blob/d3a79d15783f31982c8797d3329a423f379064e2/bert-crf4NER/bert_crf.py#L133


    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.dropout)
        self.bert_out_to_labels = nn.Linear(config.hidden_size, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        self.init_weights()
        self.bert = AutoModel.from_pretrained(config.model, config=config)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask)
        bert_output = outputs[0]
        bert_output = self.dropout(bert_output)
        probs = self.bert_out_to_labels(bert_output)
        logits = log_soft(probs, 2)
        attention_mask = attention_mask.type(torch.uint8)
        output = None
        
        if labels is not None:
            loss = -self.crf(logits, labels, mask=attention_mask, reduction='mean')
            output = loss
        
        pred_list = self.crf.decode(logits, mask=attention_mask)
        preds = torch.zeros_like(input_ids).long()
        for i, pred in enumerate(pred_list):
            preds[i, :len(pred)] = torch.LongTensor(pred)

        if output is None:
            return preds
        else:
            return (output, preds)