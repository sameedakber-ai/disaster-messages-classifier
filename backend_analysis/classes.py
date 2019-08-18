# import libraries
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from scipy import sparse
from collections import defaultdict
import pickle
import re
import cloudpickle

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

# collection of all named entities (proper nouns) in messages
all_named_entities = {'NORP':'Nationalities','FAC':'Buildings,airports,roads','ORG':'Organizations',
    'GPE':'Geo-Political Location','LOC':'Non GPE Locations','PRODUCT':'Objects,vehicles,foods','EVENT':'Named Events',
    'DATE':'Date','TIME':'Time','PERCENT':'Percentage','MONEY':'Money','QUANTITY':'Quantity'}

def tokenize(text):
    """
    Tokenize raw text string by removing punctuations, lemmatizing and removing stopwords
    """
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    tokens = word_tokenize(text)
    clean_tokens = [WordNetLemmatizer().lemmatize(w.lower()) for w in tokens if w not in stopwords.words('english')]
    return clean_tokens

def build_visualizations(df):
    """
    Build visualizations from loaded data

    INPUTS
    none

    OUTPUT
    dictionary mapping visualization labels to visualization data
    """

    print('\n\n', 'Building Visualization: Genre Counts...', '\n')
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    print('Building Visualization: Named Entities Frequency...', '\n')
    named_entities_present_related = NumberofEntitiesPresent().fit_transform(df.message[df.related==1])
    named_entities_present_non_related = NumberofEntitiesPresent().fit_transform(df.message[df.related==0])
    named_entity_data = pd.DataFrame({'Non Related': np.array(named_entities_present_related.sum(axis=0))/df[df.related==1].shape[0],
        'Related': np.array(named_entities_present_non_related.sum(axis=0))/df[df.related==0].shape[0]}, index=all_named_entities.values())

    print('Building Visualization: Category Counts...', '\n')
    categories_count = df[df.related==1].iloc[:,5:].sum(axis=0).sort_values(ascending=False)
    new_index = pd.Series(categories_count.index).str.replace('_', ' ').values
    categories_count = pd.Series(categories_count.values, index=new_index)

    print('Building Visualization: Multilabel Relation Count...', '\n')
    number_of_related = df[df.related==1].iloc[:,5:].sum(axis=1).value_counts().sort_values(ascending=False)[1:]

    print('...visualization build complete', '\n\n')

    visuals_dict = {'genre_counts': (genre_counts, genre_names), 'named_entities': named_entity_data,
    'categories_count': categories_count, 'number_of_related': number_of_related}

    cloudpickle.dump(visuals, open('visuals', 'wb'))


class MessageLengthExtractor(BaseEstimator, TransformerMixin):
    """Message length extractor for calculating length of a string of text in an array of text messages

    Attributes:
        None

    """
    def message_length(self, text):
        """Function to calculate length of text

        Args:
            text: text to calculate length of

        Returns:
            length of text
        """
        tokenized = tokenize(text)
        if tokenized:
            return len(tokenized)
        else:
            return 0

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Function to calculate lengths of texts inside an array

        Args:
            X: 1D array of text messages

        Returns:
            1D array of text lengths
        """
        lengths = pd.Series(X).apply(self.message_length)
        return lengths.values.reshape(-1,1)


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """Starting Verb Extractor class for checking existence of starting verbs in an array of text messages

    Attributes:
        None
    """

    def starting_verb(self, text):
        """Function to check if first word in text sentence is a verb

        Args:
            text: string of text

        Returns:
            True if starting verb present; False otherwise
        """
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
        """Function to transform array of texts into boolean instances depending of existence of starting verbs in text

        Args:
            X: 1D array of text messages

        Returns:
            transformed array
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return X_tagged.values.reshape(-1,1)



class IsEntityPresent(BaseEstimator, TransformerMixin):
    """Is Entity Present class to check presence of different named entities (proper nouns) in an array of text messages

    Attributes:
        None
    """

    def present_entities(self, text):
        """Function to check presence of named entity in a text

        Args:
            text: string of text

        Returns:
            list of boolean instances
        """
        text = nlp(text)
        labels = set([x.label_ for x in text.ents])
        return [1 if entity in labels else 0 for entity in all_named_entities]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """Function to apply present_entities function to each text in an array of texts

        Args:
            X: 1D array of texts

        Returns:
            transformed array
        """
        entities =  pd.Series(X).apply(self.present_entities)
        return np.array(entities.values.tolist())

class NumberofEntitiesPresent(BaseEstimator, TransformerMixin):
    """Number of Entities Present class to get the number a named entity is mentioned across an array of messages

    Attributes:
        None
    """
    def present_entities(self, text):
        """Function to count the number a named entity is mentioned in a message

        Args:
            text: string of message

        Returns:
            count of entity mentions
        """
        text = nlp(text)
        labels = [x.label_ for x in text.ents]
        total=defaultdict(int)
        for entity in all_named_entities:
            temp=0
            for label in labels:
                if entity==label:
                    temp+=1
            total[entity]+=temp
        return list(total.values())
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """Function to apply present_entities to each message in the messages array

        Args:
            X: 1D array of messages
        Returns:
            transformed array
        """
        entities =  pd.Series(X).apply(self.present_entities)
        return np.array(entities.values.tolist())