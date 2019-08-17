import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

from sklearn.base import BaseEstimator, TransformerMixin


app = Flask(__name__)

class IsEntityPresent(BaseEstimator, TransformerMixin):
    def present_entities(self, text):
        text = nlp(text)
        labels = set([x.label_ for x in text.ents])
        return [1 if entity in labels else 0 for entity in all_named_entities.keys()]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        entities =  pd.Series(X).apply(self.present_entities)
        return np.array(entities.values.tolist())

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



# load data
engine = create_engine('sqlite:///../data/disaster_database.db')
df = pd.read_sql_table('categories', engine)

# load model
model = joblib.load("../models/analyze_disaster_messages.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    class_distributions = df.iloc[:,5:]


    all_named_entities = {'NORP':'Nationalities','FAC':'Buildings, airports, highways','ORG':'Organizations',
    'GPE':'Geo-Political Location','LOC':'Non GPE Locations','PRODUCT':'Objects, vehicles, foods','EVENT':'Named Events',
    'DATE':'Date','TIME':'Time','PERCENT':'Percentage','MONEY':'Money','QUANTITY':'Quantity'}

    named_enities_present_related = IsEntityPresent().fit_transform(df.message[df.related==1])
    named_enities_present_non_related = IsEntityPresent().fit_transform(related.message[df.related==0])

    named_entity_data = pd.DataFrame({'Non Related': named_enities_present_related.sum(axis=0),
    'Related': named_enities_present_non_related.sum(axis=0)}, index=all_named_entities.values())

    categories_count = df[df.related==1].iloc[:,5:].sum(axis=0).sort_values(ascending=False)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=named_entity_data.index.tolist(),
                    y=named_entity_data['Non Related']
                ),

                Bar(
                    x=named_entity_data.index.tolist(),
                    y=named_entity_data['Related']
                )
            ],

            'layout': {
                'title': 'Distribution of Named Entities in Messages',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title': 'Named Entities'
                }
            }
        },

        {
            'data': [
                Bar(
                    x=categories_count.index.tolist(),
                    y=categories_count.values.tolist()
                )
            ],

            'layout': {
                'title': 'Category Counts for Related Messages',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title': 'categories'
                }
            }
        },

        {
            'data': [
                
            ]
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()