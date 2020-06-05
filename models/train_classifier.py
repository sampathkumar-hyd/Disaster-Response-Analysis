"""
train_classifier.py
Udacity - DSND - Disaster Resoponse Project
To run this script from models directory
> python train_classifier.py ../data/<SQLite db name> <Pickle file name>
"""

import sys

# import libraries
import pandas as pd
import numpy as np
import os
import pickle
from sqlalchemy import create_engine
import re
import nltk

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report, confusion_matrix
from sklearn.utils import parallel_backend

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def load_data(database_filepath):
    """
    load_data function
    INPUT:
        database_filepath: path to SQLite db
    OUTPUT:
        X: feature DataFrame
        Y: label DataFrame
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('dr_messages',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    return X, Y

def tokenize(text):
    """
    tokenize function
    INPUT:
        text: list of text messages (english)
    OUTPUT:
        clean_tokens: tokenized text, ready for ML modeling
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    # Split text into words using NLTK
    words = word_tokenize(text)

    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]

    # Lemmatizer
    lemmatizer = WordNetLemmatizer()
    text_tokens = []
    for tok in words:
        temp_tokens = lemmatizer.lemmatize(tok).strip()
        text_tokens.append(temp_tokens)
    return text_tokens

def build_model():
    """
    build_model function
    INPUT:
        None
    OUTPUT:
        a Scikit ML Pipeline that processes text messages and then applies a classifier.
    """
    pipeline = Pipeline([
        ('vect',TfidfVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=50)))
    ])
    # return pipeline
    parameters1 = {
        'clf__estimator__n_estimators': [10,50],
        'clf__estimator__max_features': ['auto','log2']
    }
    cv = GridSearchCV(pipeline, param_grid = parameters1 , n_jobs = -1)
    return cv

def evaluate_model(model, X_test, Y_test):
    """
    evaluate_model function    
    INPUT:
        model: Scikit ML Pipeline
        X_test: test features
        Y_test: test labels
        category_names: label names (multi-output)
    OUTPUT:
        prints out model performance (accuracy and f1score) detals
    """

    y_pred = model.predict(X_test) 
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], y_pred[:, i]))
    return

def save_model(model, model_filepath):
    """
    save_model function    
    INPUT:
        model: Scikit ML Pipeline
        model_filepath: Destination path to save .pkl file
    OUTPUT:
        None
    """

    pickle.dump(model, open(model_filepath, 'wb'))
    return

def main():
    """
    Main function for Train Classifier
    
    This function does the following:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
    """
 
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        with parallel_backend('multiprocessing'):
            print('Building model...')
            model = build_model()
            
            print('Training model...')
            model.fit(X_train, Y_train)
            
            print('Evaluating model...')
            evaluate_model(model, X_test, Y_test)

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
