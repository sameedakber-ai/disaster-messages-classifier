import sys
import pandas as pd 
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load raw message and categories data into workspace

    INPUT
    messages_filepath - system path for raw disaster message data
    categories_filepath - system path for raw disaster categories data

    OUTPUT
    pandas dataframe containing merged raw message and category data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='left')
    return df

def clean_data(df):
    """
    Clean and transform data to an analysis friendly format

    INPUT
    df - pandas dataframe

    OUTPUT
    clean pandas dataframe
    """
    categories = df.categories.str.split(';', expand=True)

    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories.columns:
        categories[column] = categories[column].str.split('-').str[1]
        categories[column] = pd.to_numeric(categories[column])

    df = pd.concat([df.drop('categories', axis=1),categories], axis=1)

    df = df[~df.duplicated()]

    df.related.where(df.related != 2, 0, inplace=True)

    # remove 'child alone' category as it contains only
    # a singular class (0). Binary classifiers need atleast
    # 2 classes for training 
    df.drop('child_alone', axis=1, inplace=True)

    return df

def save_data(df, database_filepath):
    """
    Load data onto file

    INPUT
    df - pandas dataframe
    database_filepath - file location to save data to
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql('categories', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()