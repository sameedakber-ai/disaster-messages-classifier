import sys
sys.path.insert(1, 'c:/code/Udacity/disaster_response/backend_analysis')

import json
import plotly
import pandas as pd
import pickle

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Scatter

from classes import *


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///data/disaster_database.db')
df = pd.read_sql_table('categories', engine)

# load model
model = pickle.load(open("models/analyze_disaster_messages.pkl", 'rb'))


print('Building Visualizations...')

# extract data needed for visuals
genre_counts = df.groupby('genre').count()['message']
genre_names = list(genre_counts.index)

named_enities_present_related = IsEntityPresent().fit_transform(df.message[df.related==1])
named_enities_present_non_related = IsEntityPresent().fit_transform(related.message[df.related==0])
named_entity_data = pd.DataFrame({'Non Related': named_enities_present_related.sum(axis=0),
    'Related': named_enities_present_non_related.sum(axis=0)}, index=all_named_entities.values())


categories_count = df[df.related==1].iloc[:,5:].sum(axis=0).sort_values(ascending=False)

number_of_related = df[df.related==1].iloc[:,5:].sum(axis=1).value_counts().sort_values(ascending=False)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # create visuals
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
                Scatter(
                    x=number_of_related.index.tolist(),
                    y=number_of_related.values.tolist()
                )
            ],

            'layout': {
                'title': 'Messages having Multiple Labels',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title': 'Number of Related Labels'
                }
            }
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