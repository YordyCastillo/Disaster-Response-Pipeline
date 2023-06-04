import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sqlalchemy import create_engine
import joblib



def load_data(database_filepath):
    """
    Load data from SQLite database
    
    Arguments:
    database_filepath -- string, path to SQLite database
    
    Returns:
    X -- pandas Series, feature variable
    Y -- pandas DataFrame, target variables
    category_names -- list of strings, category names
    """
    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('mycat_messages', engine)
    
    # Define feature and target variables
    X = df['message']
    Y = df.iloc[:, 4:]
    
    # Get category names
    category_names = Y.columns.tolist()
    
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize text data
    
    Arguments:
    text -- string, text data to be tokenized
    
    Returns:
    tokens -- list of strings, tokenized text data
    """
    # Normalize text data
    text = text.lower()
    
    # Tokenize text data
    tokens = word_tokenize(text)
    
    return tokens


def build_model():
    """
    Build a machine learning pipeline
    
    Returns:
    pipeline -- sklearn.pipeline.Pipeline object, machine learning pipeline
    """
    # Define the pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Define parameter grid for grid search
    parameters = {
        'clf__estimator__n_estimators': [5, 10],
        'clf__estimator__min_samples_split': [2, 4]
    }
    
    # Perform grid search to find the best parameters
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model and print classification report
    
    Arguments:
    model -- trained model object
    X_test -- pandas Series, test features
    Y_test -- pandas DataFrame, test targets
    category_names -- list of strings, category names
    """
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Print classification report for each category
    for i, category in enumerate(category_names):
        print(category)
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))
        print('------------------------')


def save_model(model, model_filepath):
    """
    Save the model as a pickle file
    
    Arguments:
    model -- trained model object
    model_filepath -- string, path to save the model
    """
    joblib.dump(model, model_filepath)

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