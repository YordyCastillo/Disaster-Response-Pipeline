import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge message and categories data from CSV files.

    Args:
        messages_filepath (str): Filepath of the messages CSV file.
        categories_filepath (str): Filepath of the categories CSV file.

    Returns:
        df (pandas.DataFrame): Merged dataframe containing message and categories data.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


import numpy as np

def clean_data(df):
    """
    Perform cleaning and preprocessing on the dataframe.

    Args:
        df (pandas.DataFrame): Dataframe containing merged message and categories data.

    Returns:
        df (pandas.DataFrame): Cleaned dataframe.
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)

    # Drop rows with 'related' category values of 2
    categories = categories.replace(2, np.nan)
    categories.dropna(subset=['related'], inplace=True)

    df.drop('categories', axis=1, inplace=True)

    df = pd.concat([df, categories], axis=1, join='inner')
    df.drop_duplicates(inplace=True)

    return df



def save_data(df, database_filename):
    """
    Save the cleaned dataframe to an SQLite database.

    Args:
        df (pandas.DataFrame): Cleaned dataframe.
        database_filename (str): Filename for the SQLite database.

    Returns:
        None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    connection = engine.connect()
    
    # Drop the table if it exists
    table_name = 'mycat_messages'
    if engine.dialect.has_table(connection, table_name):
        connection.execute(f'DROP TABLE {table_name}')
    
    # Save the DataFrame as a new table
    df.to_sql(table_name, engine, index=False)
    connection.close()


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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

