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

all_entities = ['NORP','FAC','ORG','GPE','LOC','PRODUCT','EVENT','DATE','TIME','PERCENT','MONEY','QUANTITY']

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
        return [1 if entity in labels else 0 for entity in all_entities]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        entities =  pd.Series(X).apply(self.present_entities)
        return np.array(entities.values.tolist())

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('categories', con=engine)
    category_names = df.iloc[:,5:].columns.tolist()
    X = df.message.values
    Y = df[category_names].values
    return X, Y, category_names


def build_model():

    pipeline = Pipeline([
    
        ('features', FeatureUnion([
        
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.65)),
                ('tfidf', TfidfTransformer())
            ])),

            ('entity', IsEntityPresent()),

            ('verb', StartingVerbExtractor()),

            ('length_pipeline', Pipeline([
                ('length', MessageLengthExtractor()),
                ('scalar', StandardScaler())
            ]))
        
        ])),
    
        ('clf', MultiOutputClassifier(LinearSVC(max_iter=5000), n_jobs=-1))
    
    ])


    parameters = {
            'features__text_pipeline__vect__max_df': [0.4, 0.7],
            'features__text_pipeline__vect__max_features': [7500, 12000],
            'clf__estimator__C': [0.5, 0.8],
            'features__transformer_weights':(
                {'text_pipeline': 1, 'entity': 1, 'length_pipeline': 1, 'verb': 1},
                {'text_pipeline': 1, 'entity': 0, 'length_pipeline': 0.5, 'verb': 0.5}
            )
    }


    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=4, scoring='f1_micro')

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i,cat in enumerate(category_names):
        pred = Y_pred[:,i]
        test = Y_test[:,i]
        print(cat, '\n')
        print(classification_report(test, pred, labels = np.unique(pred)))
        print('\n\n')
    pr_re_f_sup = precision_recall_fscore_support(Y_test, Y_pred, average='micro')
    print('Overall scoring metrics (micro-averged):', '\n')
    print('Precision: ', pr_re_f_sup[0], '\n')
    print('Recall: ', pr_re_f_sup[1], '\n')
    print('F1 score: ', pr_re_f_sup[2], '\n\n')

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluations = evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()