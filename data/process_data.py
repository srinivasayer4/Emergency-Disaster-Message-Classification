# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Function to load the csv files (i.e messages and categories) and merge them into a single dataframe.
    
    Parameters:
    message: csv format
        The file that contains the messages sent during disaster
    categories: csv format
        The csv file that contains the labelled output for message categories
    
    Returns:
    merged dataframe of messages and categories"""
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    return  pd.merge(messages, categories, on='id')

def clean_data(df):
    """ Function to clean the data from the provided dataframe. It will expand the categories column into 36 seperate columns and remove duplicates.
    
    Parameters:
    df: Dataframe
        The merged dataframe that has to be cleaned
    
    Returns:
    Cleaned dataframe"""
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    
    #Create name for the categories dataframe using the first row
    category_colnames = [x.split('-')[0] for x in categories.iloc[0,:]]

    # rename the columns of categories dataframe
    categories.columns = category_colnames
    
    # Converting the values in the categories dataframe into binary values
    for col_name in categories.columns:
        categories[col_name]= categories[col_name].apply(lambda x: int(x.split('-')[1]))
    
    #replacing the value 2 in 'related' column to 1
    categories['related']= categories['related'].apply(lambda x: 1 if x==2 else x)
    
    # drop the original categories column from `df`
    df= pd.merge(df, categories, left_index= True, right_index= True )
    df.drop(columns='categories', inplace= True)

    # drop duplicates
    df.drop_duplicates(inplace= True)
    return df
    
def save_data(df, database_filename):
    """Function to create a database to store the cleaned dataframe. The table name has been set to EmeregencyMessage inside the database.
    
    Parameters:
    df: Dataframe
        Cleaned dataframe that needs to be sent to the database
    database_filename: str
        File location and name of the database eg. '/data/DisasterResponse.db'
        
    Returns:
    None"""
    engine = create_engine('sqlite:///'+database_filename)
    #engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('EmergencyMessage', engine, index=False, if_exists= 'replace')

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