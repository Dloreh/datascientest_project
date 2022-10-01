import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Optional
from df_transformer import pipeline
import joblib
from fastapi import FastAPI, File, UploadFile

model = joblib.load('LR_pipe.joblib')

#class Dataset(BaseModel):
 #   reviews_modified: str
 #   nb_mots: int
 #   count_punct: int
 #   sentiment: int

api = FastAPI(
    title='Sentiment analysis'
)

@api.get('/')
def get_index():
    """Root
    """
    return {'data': 'Bonjour, vous êtes sur l\'API de Flo et JB'}

@api.get('/status')
async def get_status():
    """Returns "OK service online" if status online
    """
    return "OK service online"

def transform_data(df):
    return df.to_json(orient="records") #,media_type="application/json")

@api.post('/result', name='Post data and get result')
async def upload_data(csv_file: UploadFile = File(...)):
    """Load data and return a json file of the dataset with the result.
    """
    dataframe = pd.read_csv(csv_file.file)
    
    X = pipeline(dataframe)
    print(X)
    result = model.predict(X)
    dataframe['Resultat'] = result
    dataframe.loc[dataframe['Resultat']==1, "Resultat"] = "Positive comment"
    dataframe.loc[dataframe['Resultat']==0, "Resultat"] = "Negative comment"
    return transform_data(dataframe)
