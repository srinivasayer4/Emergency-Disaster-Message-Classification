import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import time
import nltk
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score, make_scorer
import pickle
from IPython.display import FileLink

def load_data(database_filepath):
    """ Loading the data from the database created in the process_data.py program.
    Parameters:
    database_filepath: str
        Database location e.g '../folder_name/database_name.db'
        
    Returns:
    X: Dataframe
        Contains the message column
    Y: Dataframe
        Contains 36 output columns that have the message categorization
    category_names: list
        List of output column names of Y dataframe"""
    
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    #engine = create_engine('sqlite:///{}'.format(database_filename))
    df = pd.read_sql_table('EmergencyMessage', engine, )
    
    # Splitting the dataframe into X and Y 
    X= df.message
    Y= df.iloc[:,4:]
    return X,Y, Y.columns


def tokenize(text):
    """Function to tokenize the given text.
    It removes punctuation, lowers the case, removes the stopwords, lemmatizes and stems the words in the text.
    
    Parameters:
    text: str
        The input text that needs to be tokenized
    
    Returns:
    List that is tokenized """
    
    # replace punctuations with spaces and change text to lower case
    temp = word_tokenize(re.sub(r'[\W]',' ',text).lower())
    
    # remove stop words from the sentence
    words= [x for x in temp if x not in stopwords.words('english')]
    
    # lemmatize and stem the words
    return [PorterStemmer().stem( WordNetLemmatizer().lemmatize(w)) for w in words]


def build_model():
    """ Function that builds the pipeline to tokenize the text and develops the keywords into a matrix using CountVectorizer and TFIDF transformer\
    It also uses RandomForest and Multioutput classifier to classify the message.
    
    Parameters:
    None
    
    Returns:
    cv: pipeline and GridSearch object"""
    
    # Creating the Pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer= tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    # Setting up a grid search with Random forest to find the best parameters
    parameters = {'clf__estimator__max_depth': [ None], # [200,300, None]
                 #'clf__estimator__min_samples_leaf': [1,4],
                 #'clf__estimator__min_samples_split': [2,5],
                  'clf__estimator__n_estimators': [100, 150],
                  #'tfidf__use_idf':[True, False]
                 }
    
    # Setting f1_score with micro averaging as the scorer for the gridsearch
    my_scorer = make_scorer(f1_score,average='micro' )

    cv = GridSearchCV(estimator= pipeline, param_grid= parameters, scoring= my_scorer, cv=3, verbose= 2, n_jobs= 2 )
    
    return cv                         
                         
def evaluate_model(model, X_test, y_test, category_names):
    """Function to find the F1 score for the test dataset using the trained model. It prints the necessary metrics (precision, recall and F1 score) for each ooutput column and\
    also prints the aggregate F1 score for all the columns.
    
    Parameters:
    model: pipeline gridsearch object
    X_test: Dataframe
        Dataframe with test input values
    y_test: Dataframe
        Dataframe with test output labels
    category_names: list
        List of output columns of Y dataframe.
    
    Returns:
    None"""
    #Predicting the values of X_test
    y_test_pred= model.predict(X_test)
    
    
    # Iterating through the columns to generate F1 score for each column
    for i,col in enumerate(y_test.columns):
        print(col,'\n', classification_report(y_test[col], np.transpose(y_test_pred)[i]))
        

    print('F1 Score Aggregate \nTest :', f1_score(y_test,y_test_pred, average= 'micro') )
    
    
def save_model(model, model_filepath):
    """Saves the model provided into the desired location into a pickle format.
    
    Parameters:
    model: best GridSearch model
    model_filepath: str
        Path where the model has to be stored
        
    Returns:
        None"""
    pickle.dump(model, open(model_filepath,'wb'))
    FileLink(model_filepath)


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