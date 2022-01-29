#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Edoardo Lenzi, Beatrice Portelli, Simone Scaboro'
__version__ = '1.0'
__license__ = '???'
__copyright__ = '???'

#from ade_detection.utils.logger import Logger
#LOG = Logger.getLogger(__name__)
#import zipfile
import pandas as pd
import numpy as np
from spacy.lang.en import English
import spacy
import re

# Spacy model used for tokenization and tagging
# {english, en_core_web_sm, en...}
SPACY_MODEL = "en"

class SpacyService(object):


    def __init__(self):
        spacy_model = SPACY_MODEL
        if spacy_model == 'english':
            self.nlp = English()
        else:
            self.nlp = spacy.load(spacy_model) 
        self.nlp.add_pipe(self.hashtag_pipe)
        self.nlp.add_pipe(self.html_entity_pipe)
        #self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))


    def hashtag_pipe(self, doc):
        '''Inspired by https://github.com/explosion/spaCy/issues/503
        '''
        i = 0
        while i < len(doc) - 1: 
            token = doc[i]
            if token.text == '#':
                if re.match(r'^\w+$', str(doc[i+1])):
                    with doc.retokenize() as retokenizer:
                        retokenizer.merge(doc[i:i+2])
            i += 1
        return doc

    def html_entity_pipe(self, doc):
        '''Inspired by https://github.com/explosion/spaCy/issues/503
        '''
        i = 0
        while i < len(doc) - 1: 
            token = doc[i]
            if token.text == '&':
                if re.match(r'^\w+$', str(doc[i+1])) and str(doc[i+2])==";":
                    with doc.retokenize() as retokenizer:
                        retokenizer.merge(doc[i:i+3])
            i += 1
        return doc

    def tokenizer(self, text:str):
        return self.nlp(text)


    def sentencizer(self, text:str):
        return self.nlp(text).sents