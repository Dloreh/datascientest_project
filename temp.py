import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import joblib
import string

#NLTK
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

#wordcloud
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#Sickit learn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import  make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


#### Commentaire
df = pd.read_csv('https://assets-datascientest.s3-eu-west-1.amazonaws.com/de/total/reviews.csv', delimiter=',')
df.head()
df.info()
#### Check valeurs uniques dans variable Rating
#check df_rating = uniquement des nombres
df['Rating'].unique() #OK
#### Remplacement des valeurs 'missing' dans Year_month
#check Year_Month valeurs uniques
df['Year_Month'].unique()

#Nettoyage Year_month
df.loc[df['Year_Month']=='missing', 'Year_Month'] = np.nan

#check Year_Month valeurs uniques
df['Reviewer_Location'].unique()
df[df['Reviewer_Location'] == 'missing']

## PROCESSING TEXT REVIEW
### LOWER TEXT
df['reviews_modified'] = df['Review_Text'].str.lower()
### RETRAIT PONCTUATION
PUNCT_TO_REMOVE = string.punctuation

def remove_punctuation(text):
    """Fonction pour retirer la ponctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

df['reviews_modified'] = df['reviews_modified'].apply(lambda text : remove_punctuation(text))    
### LEMMATISATION
# TOKENIZATION
def lemmatize_text(text):
    """Fonction pour lemmatiser le texte"""
    return [WordNetLemmatizer().lemmatize(e) for e in WhitespaceTokenizer().tokenize(text)]

df['reviews_modified'] = df['reviews_modified'].apply(lemmatize_text)
df['reviews_modified'] = df['reviews_modified'].apply(lambda x : " ".join(x))
### CALCUL NOMBRE DE MOTS
#Création d'une variable contenant le nombre de mots de cette nouvelle variable afin d'obtenir la longueur du commentaire brut.
df['nb_mots'] = df['Review_Text'].str.split(" ").str.len()
#visualisation du nombre de mots moyens par rating
df.groupby('Rating').mean().reset_index()

### CALCUL PONCTUATION
df['count_punct'] = df["Review_Text"].apply(lambda s: sum([1 for x in s if x in '?!']))
## DATA EXPLORATION
#visualisation du nombre d emots moyens par rating via Barplot
fig = px.bar(df.groupby('Rating').mean().reset_index(), x='Rating', y='nb_mots', width=800, height=400, title='Nombre de mots moyen par ratings')

fig.update_layout(
            title={
            'x':0.5,
            'xanchor': 'center',
        })

fig.show()
#visualisation du nombre de mots par rating via boxplot
fig = px.box(df, x='Rating', y='nb_mots', width=800, height=400, title='Box plot nombre de mots VS rating')

fig.update_layout(
            title={
            'x':0.5,
            'xanchor': 'center',
        })

fig.show()
#check d'une valeur potentiellement abbérante
df.loc[(df['Rating'] == 4) & (df['nb_mots']> 4000)]

#filtre df sur valeur nb de mots <2000
df_viz = df[df['nb_mots']<2000]
# Nouvelle visuation des boxpliot post traitement.

fig = px.box(df_viz, x='Rating', y='nb_mots', width=800, height=400, title='Box plot nombre de mots VS rating post retraitement')

fig.update_layout(
            title={
            'x':0.5,
            'xanchor': 'center',
        })

fig.show()
#visualisation du nombre d emots moyens par rating via Barplot
fig = px.bar(df.groupby('Rating').mean().reset_index(), x='Rating', y='nb_mots', width=800, height=400, title='Nombre de mots moyen par ratings post retraitement')

fig.update_layout(
            title={
            'x':0.5,
            'xanchor': 'center',
        })

fig.show()
df_viz.info()
### Nombre de commentaires par pays
df_count_by_country = pd.DataFrame(df['Reviewer_Location'].value_counts()).reset_index()
fig = px.bar(df_count_by_country, x='index', y=['Reviewer_Location'])
fig.show()


### Modèle classification analyse du sentiement
#### Retraitement des stop words
wc_positive = df[df['Rating']>3]
wc_negative = df[df['Rating']<3]
wc_neutre = df[df['Rating']==3]
fig = px.bar(pd.DataFrame(df['Rating'].value_counts()).reset_index(), x='index', y='Rating', title="Nombre de commentaires par Ratings")
fig.show()


## WORD CLOUD VIZ
#### GoW ON NEGATIVE
stop_words = set(stopwords.words('english'))

# Start with one review:
text = ''
for e in wc_negative['reviews_modified']:
    text += e

wc = WordCloud(background_color="black", max_words=30, stopwords=stop_words, max_font_size=50, random_state=42)

plt.figure(figsize= (10,6)) # Initialisation d'une figure
wc.generate(text)           # "Calcul" du wordcloud
plt.imshow(wc) # Affichage
plt.show()
### GoW ON POSITIBVE
stop_words = set(stopwords.words('english'))

# Start with one review:
text = ''
for e in wc_positive['reviews_modified']:
    text += e

wc = WordCloud(background_color="black", max_words=30, stopwords=stop_words, max_font_size=50, random_state=42)

plt.figure(figsize= (10,6)) # Initialisation d'une figure
wc.generate(text)           # "Calcul" du wordcloud
plt.imshow(wc) # Affichage
plt.show()
### GoW ON NEUTRAL
stop_words = set(stopwords.words('english'))

# Start with one review:
text = ''
for e in wc_neutre['reviews_modified']:
    text += e

wc = WordCloud(background_color="black", max_words=30, stopwords=stop_words, max_font_size=50, random_state=42)

plt.figure(figsize= (10,6)) # Initialisation d'une figure
wc.generate(text)           # "Calcul" du wordcloud
plt.imshow(wc) # Affichage
plt.show()

df['sentiment'] = 1
df.loc[df['Rating']<4, 'sentiment'] = 0
df[df['Rating']<4]['sentiment'].unique()
### GoW on whole reviews
stop_words = set(stopwords.words('english'))

# Start with one review:
text = ''
for e in df['reviews_modified']:
    text += e

wc = WordCloud(background_color="black", max_words=30, stopwords=stop_words, max_font_size=50, random_state=42)

plt.figure(figsize= (10,6)) # Initialisation d'une figure
wc.generate(text)           # "Calcul" du wordcloud
plt.imshow(wc) # Affichage
plt.show()
### MODEL & PIPELINE
#SKLEARN
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
#STRING
import string
#NLTK
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
    

def pipeline(vectorizer, model, df):
    """Fonction pipeline modèle NLP"""

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

    #Calcul nombre de mots
    df['nb_mots'] = calculate_nb_of_words(df, 'reviews_modified')

    df['punct_nb'] = punct_to_count(df, 'Review_Text', '?!')


    if model == 'DecisionTreeClassifier':
        textual_transformer = Pipeline(steps=[('vectorizer', vectorizer)])
        preprocessor = ColumnTransformer(transformers=[("text", textual_transformer, "reviews_modified"),], remainder="passthrough")
        # Ici il faut préciser le nom de ta variable textuelle à transformer à la place de "reviews_modified" si elle est différente
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", DecisionTreeClassifier())])
        return pipe

    if model =='LogisticRegression':
        textual_transformer = Pipeline(steps=[('vectorizer', vectorizer)])
        preprocessor = ColumnTransformer(transformers=[("text", textual_transformer, "reviews_modified"), ])
        # Ici il faut préciser le nom de ta variable textuelle à transformer à la place de "reviews_modified" si elle est différente
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", LogisticRegression())])
        return pipe

    if model == 'GradientBoostingClassifier':
        textual_transformer = Pipeline(steps=[('vectorizer', vectorizer)])
        preprocessor = ColumnTransformer(transformers=[("text", textual_transformer, "reviews_modified"), ])
        # Ici il faut préciser le nom de ta variable textuelle à transformer à la place de "reviews_modified" si elle est différente
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", GradientBoostingClassifier())])
        return pipe



df.info()
#Train test split
X_train, X_test, y_train, y_test = train_test_split(df[['reviews_modified', 'nb_mots', 'count_punct']], df['sentiment'], test_size=0.2, shuffle=True)
### LOGISTIC REGRESSION
pipe = pipeline(CountVectorizer(stop_words='english', max_features=10000), 'LogisticRegression', df)
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print(f"L'accuracy score est : {accuracy_score(y_test, y_pred)} et le f1 score est : {f1_score(y_test, y_pred)}")
confusion_matrix(y_test, y_pred)/len(y_test)*100
tn, fp, fn, tp = (confusion_matrix(y_test, y_pred)/len(y_test)*100).ravel()
import joblib

joblib.dump(pipe, 'LR_pipe.joblib')
### GRADIENT BOOSTING CLASSIFIER
pipe = pipeline(CountVectorizer(stop_words='english', max_features=10000), 'GradientBoostingClassifier', df)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print(f"L'accuracy score est : {accuracy_score(y_test, y_pred)} et le f1 score est : {f1_score(y_test, y_pred)}")
confusion_matrix(y_test, y_pred)/len(y_test)*100
tn, fp, fn, tp = (confusion_matrix(y_test, y_pred)/len(y_test)*100).ravel()
### DECISION TREE CLASSIFIER
pipe = pipeline(CountVectorizer(stop_words='english', max_features=10000), 'DecisionTreeClassifier', df)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print(f"L'accuracy score est : {accuracy_score(y_test, y_pred)} et le f1 score est : {f1_score(y_test, y_pred)}")
confusion_matrix(y_test, y_pred)/len(y_test)*100
tn, fp, fn, tp = (confusion_matrix(y_test, y_pred)/len(y_test)*100).ravel()
### import model test
model = joblib.load('LR_pipe.joblib')
