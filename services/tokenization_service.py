#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Edoardo Lenzi, Beatrice Portelli, Simone Scaboro'
__version__ = '1.0'
__license__ = '???'
__copyright__ = '???'


from pandas.core.frame import DataFrame
from tqdm.auto import tqdm
from os import path
import html

from services.spacy_service import SpacyService
import re

from domain.token import Token

class TokenizationService(object):

    '''Tokenization script
    '''

    def __init__(self):  
        self.spacy = SpacyService()
    
    def tokenize_and_clean(self, docs):
        #print('Tokenization in progress...')
        doc_tokens_list = []
        for doc in tqdm(docs, desc="tokenizing documents"):

            tokens = self.tokenizer(doc)
            doc_tokens = self.tokens_postprocessing(tokens)
            doc_tokens_list.append(doc_tokens)
            
        #print('Tokenization completed successfully!')
        return doc_tokens_list

    def nearest(self, array, value):
        if len(array) > 0:
            winner = array[0]
            best_delta = abs(array[0] - value)
            for i in array:
                delta = abs(i - value)
                if delta < best_delta:
                    winner = i 
            return winner
        else:
            return None


    def tokenizer(self, text:str):
        tokens = []
        for t in self.spacy.tokenizer(text):
            tokens.append(Token(begin = t.idx, 
                                end = t.idx + len(t.text), 
                                text = t.text))
        return tokens


    def tokens_postprocessing(self, tokens):
        for t in tokens:
            if self.is_number(t.text):
                t.text = 'number'
            elif self.is_hashtag(t.text):
                t.text = t.text[1:len(t.text)]
            elif self.is_username(t.text):
                t.text = 'username'
            elif self.is_link(t.text):
                t.text = 'link'
            elif self.is_html_entity(t.text):
                t.text = html.unescape(t.text)
            else:
                t.text = self.delete_triples(t.text)
        return tokens

    def is_html_entity(self, token):
        return token[0] == '&' and token[-1] == ";" and re.match(r'^\w+$', token[1:-1])

    def is_number(self, token):
        return token.replace('.','',1).replace(',','',1).isdigit()


    def is_hashtag(self, token):
        if len(token) > 0:
            return token[0] == '#' and len(token) > 1
        else:
            return False

    def is_username(self, token):
        if len(token) > 0:
            return token[0] == '@' and len(token) > 1
        else:
            return False


    def is_link(self, token):
        if len(token) > 4:
            return token[0:4] == 'http' or token[0:3] == 'www' 
        else:
            return False


    def delete_triples(self, token):
        i = 0
        token = list(token)
        while i < len(token) - 2:
            if token[i] == token[i+1] and token[i+1] == token[i+2] and token[i].isalpha():
                del(token[i])
                i -= 1   
            i += 1            
        return "".join(token)
