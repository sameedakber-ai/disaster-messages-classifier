import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from scipy import sparse
from collections import defaultdict
import pickle
import re

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download(['stopwords', 'punkt', 'averaged_perceptron_tagger', 'wordnet'])

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV


all_named_entities = {'NORP':'Nationalities','FAC':'Buildings, airports, highways','ORG':'Organizations',
    'GPE':'Geo-Political Location','LOC':'Non GPE Locations','PRODUCT':'Objects, vehicles, foods','EVENT':'Named Events',
    'DATE':'Date','TIME':'Time','PERCENT':'Percentage','MONEY':'Money','QUANTITY':'Quantity'}

def tokenize(text):
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    tokens = word_tokenize(text)
    clean_tokens = [WordNetLemmatizer().lemmatize(w.lower()) for w in tokens if w not in stopwords.words('english')]
    return clean_tokens

class MessageLengthExtractor(BaseEstimator, TransformerMixin):

        def message_length(self, text):
            tokenized = tokenize(text)
            if tokenized:
                return len(tokenized)
            else:
                return 0

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            lengths = pd.Series(X).apply(self.message_length)
            return lengths.values.reshape(-1,1)


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            tokenized = tokenize(sentence)
            if tokenized:
                pos_tags = nltk.pos_tag(tokenized)
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return X_tagged.values.reshape(-1,1)



class IsEntityPresent(BaseEstimator, TransformerMixin):


    def present_entities(self, text):
        text = nlp(text)
        labels = set([x.label_ for x in text.ents])
        return [1 if entity in labels else 0 for entity in all_named_entities]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        entities =  pd.Series(X).apply(self.present_entities)
        return np.array(entities.values.tolist())