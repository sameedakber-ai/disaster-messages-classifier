import sys
import os
classes_path = os.path.join(os.getcwd(), 'backend_analysis')
sys.path.insert(1, classes_path)
from classes import *

import cloudpickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('categories', con=engine)
    category_names = df.iloc[:,4:].columns.tolist()
    X = df.message.values
    Y = df[category_names].values
    return X, Y, category_names


def build_model():

    pipeline = Pipeline([
    
        ('features', FeatureUnion([
        
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('entity', IsEntityPresent()),

            ('verb', StartingVerbExtractor()),

            ('length_pipeline', Pipeline([
                ('length', MessageLengthExtractor()),
                ('scalar', StandardScaler())
            ]))
        
        ])),
    
        ('clf', MultiOutputClassifier(LinearSVC(max_iter=5000)))
    
    ])


    parameters = {
            'features__text_pipeline__vect__max_df': [0.4, 0.65],
            'features__text_pipeline__vect__max_features': [7500, 12000],
            'clf__estimator__C': [0.5, 0.8],
            'features__transformer_weights':(
                {'text_pipeline': 1, 'entity': 1, 'length_pipeline': 1, 'verb': 1},
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
    cloudpickle.dump(model, open(model_filepath, "wb"))


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