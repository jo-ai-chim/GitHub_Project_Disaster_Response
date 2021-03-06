import sys

import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base

from textblob import TextBlob
from time import sleep

import pandas as pd
import numpy as np

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
    messages_filepath - filepath to the csv file with disaster response messages 
    categories_filepath - filepath to the csv file with corresponding categories for the disaster response messages 

    OUTPUT:
    df - dataframe with the two files merged
   
    This function reads in two csv files and merges them based on the column id
    '''
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on=['id'])
    
    return df


def clean_data(df):
    '''
    INPUT:
    df - dataframe with the messages and categories 
    
    OUTPUT:
    df - cleaned dataframe with the messages and categories 
   
    This function cleans the dataframe by splitting the categories column and transforming it into numeric values. Afterwards it removes duplicates.
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = [row[i].partition("-")[0] for i in range(36)]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, sort=False)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Add Language Column
    df_clean['language'] = np.nan
    df_clean = df_clean.astype({'language': 'object'})
    for index, row in df_clean.iterrows():
        if len(str(row['original'])) >= 3:
            sleep(0.4)
            var = TextBlob(str(row['original'])).detect_language()
            df_clean.set_value(index,'language',var)
        else:
            df_clean.set_value(index,'language','diverse')
    
    # Fill the not identifiable languages with 'div'
    df_clean = df_clean.fillna(value={'language': 'diverse'})
    
    return df 


def save_data(df, database_filename):
    '''
    INPUT:
    df - cleaned dataframe with the messages and categories 
    database_filename - Name of the database the cleaned data should be stored in

    This function saves the clean dataset into an sqlite database by using pandas to_sql method combined with the SQLAlchemy library.
    '''  
    
    engine = create_engine('sqlite:///' + database_filename)
    Base = declarative_base()
    Base.metadata.drop_all(engine)
    df.to_sql('messages_with_cat', engine, index=False)      


def main():
    '''
    This is the main function which is executed when calling the process_data.py file over the console. It reads in the arguments 
    
        - messages_filepath
        - categories_filepath
        - database_filepath
    
    and executes the functions above. If one argument is missing an error is raised.
    '''
    
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
