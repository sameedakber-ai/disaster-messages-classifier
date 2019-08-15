import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import re
import nltk
nltk.download(['stopwords', 'punkt', 'averaged_perceptron_tagger', 'wordnet'])
from scipy import sparse
from collections import defaultdict
import pickle

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('categories', con=engine)
    columns_not_for_analysis = ['related', 'child_alone', 'id', 'original', 'genre']
    df.drop(columns_not_for_analysis, axis=1, inplace=True)
    X = df.message.values
    Y = df.drop('message', axis=1).values
    category_names = df.drop('message', axis=1).columns.tolist()

    return X, Y, category_names


def tokenize(text):
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    tokens = word_tokenize(text)
    clean_tokens = [WordNetLemmatizer().lemmatize(w.lower()) for w in tokens if w not in stopwords.words('english')]
    return clean_tokens


def build_model():

    class MessageLengthExtractor(BaseEstimator, TransformerMixin):
        def message_length(self, text):
            return len(tokenize(text))

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



    pipeline = Pipeline([
    
        ('features', FeatureUnion([
        
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        
            ('verb', StartingVerbExtractor()),
        
            ('length', MessageLengthExtractor())
        
        ])),
    
        ('clf', MultiOutputClassifier(LinearSVC(class_weight='balanced', dual=True), n_jobs=-1))
    
    ])



    parameters = {
            'features__text_pipeline__vect__ngram_range': [(1, 1), (1, 2)],
            'features__text_pipeline__vect__max_df': [0.5, 0.75, 1.0],
            'features__text_pipeline__vect__max_features': [500, 5000, 10000],
            'features__text_pipeline__tfidf__use_idf': [True, False],
            'clf__estimator__C': [0.1, 0.5, 1],
            'features__transformer_weights':(
                {'text_pipeline': 1, 'verb': 1, 'length': 1},
                {'text_pipeline': 1, 'verb': 0.5, 'length': 0.5},
            )
    }


    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=3, scoring='f1_weighted')

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    evaluations = defaultdict(str)
    for i,cat in enumerate(category_names):
        pred = Y_pred[:,i]
        test = Y_test[:,i]
        evaluations[cat] = classification_report(test, pred, labels = np.unique(pred), output_dict=True)
    return evaluations

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
        evaluate_model(model, X_test, Y_test, category_names)

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