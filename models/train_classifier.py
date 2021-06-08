import sys
# import main libraries 
import pandas as pd
import numpy as np

#import text libraries
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet') # download for lemmatization
from sqlalchemy import create_engine
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.datasets import make_multilabel_classification
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion

import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("Data_cleaned", engine)
    X = df['message']
    Y = df.drop(['message', 'genre'], axis=1)
    columns = Y.columns
    
    return X,Y,columns
    


def tokenize(text):
    # ELIMINAMOS LOS ESPACIOS, TRANSFORMAMOS EN LOWERCASE
    #utilizamos el paquete re para eliminar todo lo que no sea [^a-zA-Z0-9]
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip()) 
    #tokenizamos el texto
    tokens = text.split()
    #eliminamos aquellos tokens que son stopwords
    words = [w for w in tokens if w not in stopwords.words("english")]
    #transformamos las palabras en su raíz, para evitar derivaciones
    lemmed_tokens = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmed_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(random_state=42)))
])
    
#     parameters = {
#         'clf__estimator__criterion': ['gini','entropy'],
#         'clf__estimator__max_depth':[ None, 3,4,5],
#         'clf__estimator__max_features': ['auto','sqrt'],
#         'clf__estimator__n_estimators': [3,5,10],
#         'clf__estimator__n_jobs': [-1],
#         'clf__estimator__verbose': [0,1,2,3],
#     }

#     cv = GridSearchCV(pipeline, param_grid = parameters, n_jobs = 12)
    
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
#     for count in range(len(y_pred[0,:])):
#         print(category_names[count],classification_report(Y_test[:,count],y_pred[:,count]), sep="\n") 
        
def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
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