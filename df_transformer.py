import pandas as pd
#NLTK
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import numpy as np
import string 



def pipeline(df):
    """Fonction pipeline modèle NLP"""

    df['reviews_modified'] = df['Review_Text'].str.lower()
    df['sentiment'] = 1
    df.loc[df['Rating']<4, 'sentiment'] = 0

    def calculate_nb_of_words(df, column):
        """Fonction pour calculer le nombre de mots"""
        df['nb_mots'] = df[column].str.split(" ").str.len()
        return df['nb_mots']

    def punct_to_count(df, column, ponctuation_str):
        """Fonction pour calculer le nombre par ponctuation"""
        df['count_punct'] = df[column].apply(lambda s: sum([1 for x in s if x in ponctuation_str]))
        return df['count_punct']

    def remove_punctuation(text):
        """Fonction pour retirer la ponctuation"""
        PUNCT_TO_REMOVE = string.punctuation
        return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

    def lemmatize_text(text):
        """Fonction pour lemmatiser le texte"""
        return [WordNetLemmatizer().lemmatize(e) for e in WhitespaceTokenizer().tokenize(text)]

    def text_tranformation(df, column): 
        """Fonction pour transformer le texte""" 
        df['text_modified']  = df[column].str.lower()
        df['text_modified'] = df['text_modified'].apply(lambda text : remove_punctuation(text))
        df['text_modified'] = df['text_modified'].apply(lemmatize_text)
        df['text_modified'] = df['text_modified'].apply(lambda x : " ".join(x))
        return df['text_modified']

    #Text preprocessing
    df['reviews_modified'] = text_tranformation(df, 'Review_Text')
    print("################### Text prepro")
    print(df.info())   

    #Calcul nombre de mots
    df['nb_mots'] = calculate_nb_of_words(df, 'reviews_modified')
    print("################### Calcul nb mot")
    print(df.info()) 

    df['punct_nb'] = punct_to_count(df, 'Review_Text', '?!')
    print("################### Punct")
    print(df.info()) 

    #df = df[['reviews_modified', 'nb_mots', 'count_punct','sentiment']]
    X= df[['reviews_modified', 'nb_mots', 'count_punct']]
    #y= df['sentiment']
    return X #df, X, y

def text_transform(text):
    """Fonction pipeline modèle NLP"""

    df = pd.DataFrame({'Review_Text': [text]})
    df['reviews_modified'] = df['Review_Text'].str.lower()

    def calculate_nb_of_words(df, column):
        """Fonction pour calculer le nombre de mots"""
        df['nb_mots'] = df[column].str.split(" ").str.len()
        return df['nb_mots']

    def punct_to_count(df, column, ponctuation_str):
        """Fonction pour calculer le nombre par ponctuation"""
        df['count_punct'] = df[column].apply(lambda s: sum([1 for x in s if x in ponctuation_str]))
        return df['count_punct']

    def remove_punctuation(text):
        """Fonction pour retirer la ponctuation"""
        PUNCT_TO_REMOVE = string.punctuation
        return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

    def lemmatize_text(text):
        """Fonction pour lemmatiser le texte"""
        return [WordNetLemmatizer().lemmatize(e) for e in WhitespaceTokenizer().tokenize(text)]

    def text_tranformation(df, column): 
        """Fonction pour transformer le texte""" 
        df['text_modified']  = df[column].str.lower()
        df['text_modified'] = df['text_modified'].apply(lambda text : remove_punctuation(text))
        df['text_modified'] = df['text_modified'].apply(lemmatize_text)
        df['text_modified'] = df['text_modified'].apply(lambda x : " ".join(x))
        return df['text_modified']

    #Text preprocessing
    df['reviews_modified'] = text_tranformation(df, 'Review_Text')
    print("################### Text prepro")
    print(df.info())   

    #Calcul nombre de mots
    df['nb_mots'] = calculate_nb_of_words(df, 'reviews_modified')
    print("################### Calcul nb mot")
    print(df.info()) 

    df['punct_nb'] = punct_to_count(df, 'Review_Text', '?!')
    print("################### Punct")
    print(df.info()) 

    X= df[['reviews_modified', 'nb_mots', 'count_punct']]
    return X

