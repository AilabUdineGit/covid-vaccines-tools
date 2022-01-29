#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Edoardo Lenzi, Beatrice Portelli'
__version__ = '1.0'
__license__ = '???'
__copyright__ = '???'

from domain.document import Document
from domain.subtoken import SubToken

from torch.utils.data import TensorDataset
from torch import LongTensor

class BaseTaskLoader(object):

    
    # @OK
    def __init__(self, max_seq_len, annotation_type):
        self.annotation_type = annotation_type
        self.max_seq_len = max_seq_len
        

    # @OK
    def numericalize(self, docs, tokenizer):
        for doc in docs:
            CLS = tokenizer.cls_token
            PAD = tokenizer.pad_token
            SEP = tokenizer.sep_token

            doc.subtokens.insert(0, SubToken(None, CLS))
            doc.tags.insert(0, 'O')
            doc.subtokens.append(SubToken(None, SEP))
            doc.tags.append('O')
            
            doc.subtokens.extend([SubToken(None, PAD)] * (self.max_seq_len - len(doc.subtokens)))
            doc.tags.extend(['O'] * (self.max_seq_len - len(doc.tags)))
            doc.attention_mask = [0 if x.text == PAD else 1 for x in doc.subtokens]
            doc.num_subtokens = tokenizer.convert_tokens_to_ids([x.text for x in doc.subtokens])
            doc.num_tags = self.convert_iob_tags_to_ids(doc.tags, doc.subtokens, tokenizer) 
            assert len(doc.subtokens) == len(doc.tags)
            assert len(doc.attention_mask) == len(doc.tags)
            assert len(doc.num_tags) == len(doc.tags)
            assert len(doc.num_subtokens) == len(doc.subtokens)
        return docs


    def convert_iob_tags_to_ids(self, tags, subtokens, tokenizer):
        num_tags = []
        for i, t in enumerate(tags):
            if subtokens[i].text == tokenizer.pad_token:
                num_tags.append(-1)
            else:
                if t == 'O':
                    num_tags.append(0)
                else:
                    annotation = t[2:]
                    index = self.index_by_annotation(annotation)
                    if t[0] == 'B':
                        num_tags.append(index * 2 + 2)
                    elif t[0] == 'I':
                        num_tags.append(index * 2 + 1)
        return num_tags


    def index_by_annotation(self, annotation):
        return 0


    # def find_split_index(self, sdoc):
    #     max_seq_len = self.max_seq_len - 2
    #     split_candidate = sdoc.subtokens[max_seq_len]
    #     ends = list(filter(lambda x: x <= split_candidate.token.end, [x.end for x in sdoc.doc.sentences]))
    #     end = max(ends) if len(ends) > 0 else split_candidate.token.end 
    #     subtoken_index = self.nearest_subtoken(sdoc.subtokens, end)
    #     # TODO
    #     if max_seq_len >= subtoken_index and subtoken_index > 5: #and max_seq_len >= (len(sdoc.subtokens) - subtoken_index):
    #         pass
    #     else:
    #         print("@@@@ fixing")
    #         subtoken_index = max_seq_len
    #     
    #     if sdoc.doc.external_id == "1009625926571704321" and subtoken_index == 0:
    #         print("@@@@@@@@@@@@@@@")
    #         
    #         print("max_seq_len", max_seq_len)
    #         print("split_candidate", split_candidate.text)
    #         
    #         print(sdoc.doc.external_id)
    #         print(sdoc.text)
    #         
    #         print([t.text for t in sdoc.subtokens])
    #         #print(sdoc.tags)
    #         
    #         print(ends)
    #         print(end)
    #         print("---------------")
    #         print(subtoken_index)
    #         print("@@@@@@@@@@@@@@@")
    #         #exit()
    #     
    #     return (sdoc.subtokens[subtoken_index].token.end, subtoken_index)


    # def nearest_subtoken(self, array, value):
    #         winner = 0
    #         best_delta = abs(array[0].token.end - value)
    #         for i, s in enumerate(array):
    #             delta = abs(s.token.end - value)
    #             if delta <= best_delta:
    #                 best_delta = delta
    #                 winner = i 
    #         return winner


    # def subtokens_biluo_tagging(self, sdoc, task):
    #     tags = ['O'] * len(sdoc.subtokens) 
    #     for span in sdoc.doc.spans:
    #         for annotation_type in task.goal:
    #             if span.contains_annotation(annotation_type):
    #                 begin = span.tokens[0].subtokens_interval[0]
    #                 end = span.tokens[-1].subtokens_interval[1]
    #                 if begin == end - 1:
    #                     tags[begin] = 'U-' + annotation_type.name
    #                 else:
    #                     tags[begin] = 'B-' + annotation_type.name 
    #                     tags[end - 1] = 'L-' + annotation_type.name 
    #                     for i in range(begin + 1, end - 1):
    #                         tags[i] = 'I-' + annotation_type.name
    #     sdoc.tags = tags
    #     return sdoc


    # def tokens_single_biluo_tagging(self, sdoc, task):
    #     tags = ['O'] * len(sdoc.doc.tokens) 
    #     for span in sdoc.doc.spans:
    #         for annotation_type in task.goal:
    #             if span.contains_annotation(annotation_type):
    #                 begin = sdoc.doc.tokens.index(span.tokens[0])
    #                 end = sdoc.doc.tokens.index(span.tokens[-1])
    #                 if begin == end:
    #                     tags[begin] = 'U'
    #                 else:
    #                     tags[begin] = 'B'
    #                     tags[end] = 'L'
    #                     for i in range(begin + 1, end):
    #                         tags[i] = 'I'
    #     sdoc.tags = tags
    #     return sdoc


    # def biluo_to_iob(self, biluo):
    #     for i, t in enumerate(biluo):
    #         if t[0:1] == 'U':
    #             biluo[i] = t.replace('U', 'B', 1)
    #         elif t[0:1] == 'L':
    #             biluo[i] = t.replace('L', 'I', 1)
    #     return biluo 


    # def biluo_to_io(self, biluo):
    #     for i, t in enumerate(biluo):
    #         if t[0:1] == 'U':
    #             biluo[i] = t.replace('U', 'I', 1)
    #         elif t[0:1] == 'L':
    #             biluo[i] = t.replace('L', 'I', 1)
    #         elif t[0:1] == 'B':
    #             biluo[i] = t.replace('B', 'I', 1)
    #     return biluo 


    # @OK!
    def subtokenize(self, doc, tokenizer):
        subtokens = []
        for token in doc.tokens:
            for chunk in tokenizer.tokenize(token.text):
                subtoken = SubToken(token, chunk)
                subtokens.append(subtoken)
                token.subtokens.append(subtoken)
        doc.subtokens = subtokens
        return doc
    

class BertTaskLoader(BaseTaskLoader):

    def __init__(self, tokenizer, docs, max_seq_len, annotation_type="IOB"):
        super(BertTaskLoader, self).__init__(max_seq_len, annotation_type)
        self.tokenizer = tokenizer
        self.documents = self.load(docs)

    def load(self, docs):
        i = 0
        while i < len(docs):
            doc = docs[i]
            if doc.subtokens == None:
                doc = self.subtokenize(doc, self.tokenizer)
                # doc = self.subtokens_biluo_tagging(doc, self.task)
                # if self.task.notation == NOTATION.IOB:
                #     doc.tags = self.biluo_to_iob(doc.tags)
                # if self.task.notation == NOTATION.IO:
                #     doc.tags = self.biluo_to_io(doc.tags)   
                doc.tags = ["O" for _ in doc.subtokens]   # simply O, just to fill it
            if len(doc.subtokens) > self.max_seq_len - 2: 
                # new_doc = doc.copy()
                # new_doc.id = docs[-1].id + 1 
                # (char_index, subtoken_index) = self.find_split_index(doc) 
                # doc.subtokens = doc.subtokens[:subtoken_index]
                # new_doc.subtokens = new_doc.subtokens[subtoken_index:]
                # doc.tags = doc.tags[:subtoken_index]
                # new_doc.tags = new_doc.tags[subtoken_index:]
                # doc.text = doc.text[:char_index]
                # new_doc.text = new_doc.text[char_index:]
                # docs.append(new_doc)
                doc.subtokens = doc.subtokens[:self.max_seq_len-2]  # simply crop
                doc.tags = doc.tags[:self.max_seq_len-2]            # simply crop
                assert len(doc.subtokens) <= self.max_seq_len - 2
            i += 1
        docs = self.numericalize(docs, self.tokenizer)
        return docs
    
    def get_tensor_dataset(self):
        return TensorDataset( LongTensor([x.num_subtokens for x in self.documents]),
                              LongTensor([x.attention_mask for x in self.documents]),
                              LongTensor([x.num_tags for x in self.documents]),
                              LongTensor([x.id for x in self.documents]) )