import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import sqlite3
from sqlalchemy import create_engine

import pickle

import re
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    '''
    INPUT:
    database_filepath - filepath to the sqllite database where the data from the etl pipeline is stored file with disaster response messages 
    
    OUTPUT:
    X - dataframe with the features
    Y - dataframe with the labels
    category_names - list of the possible categories
   
    This function reads in the train and test data from the sqlite database and splits them in features and labels
    '''
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages_with_cat', engine)
    X = df['message']
    Y = df.drop(['id','message','original', 'genre'], axis=1)
    category_names = [column for column in Y]
    return X, Y, category_names

def tokenize(text):
    '''
    INPUT:
    text - text which should be tokenized 
    
    OUTPUT:
    clean_tokens - tokens returned 
    
    This function tokenizes the text from the features
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    This class describes an Transformer to create a new feature out of an given text feature. It returns true if the txt begins with an verb and false if not.
    '''
    
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    '''
    OUTPUT:
    cv - GridSearchCV object
    
    This function specifies the pipelien for the model and the paramters which should be considered for optimizing the model and then creates a corresponding GridSearchCV object 
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False)
        }

    # create grid search object
    #cv = GridSearchCV(pipeline, param_grid=parameters)
    cv = pipeline
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    model - the model which should be evaluated 
    X_test - features from the test data
    Y_test - labels from the test data
    category_names - list of the possible categories
    
    OUTPUT:
    prints the classification_report for each category 
    
    This function evaluates the given model with the test data and prints the classification report for each category
    '''
    
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred)
    
    labels = np.unique(Y_pred)
    print("Labels:", labels)
    for i in range(len(category_names)):
        Y_test_temp = Y_test.iloc[:,i].values.tolist()
        Y_pred_temp = Y_pred[i].values.tolist()
        print("Classification Report " + Y_test.columns[i] + ":\n" + classification_report(Y_test_temp, Y_pred_temp, labels=labels))
    print("\nBest Parameters:", model.best_params_)
    

def save_model(model, model_filepath):
    '''
    INPUT:
    model - the model which should be saved 
    model_filepath - path to the file the model should be saved in
    
    This function saves the model to the given path
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    This is the main function which is executed when calling the train_classifier.py file over the console. It reads in the arguments 
    
        - database_filepath
        - model_filepath
    
    and executes the functions above. If one argument is missing an error is raised.
    '''
    
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
