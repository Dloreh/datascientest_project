import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Optional
from df_transformer import pipeline, text_transform
import joblib
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

#Setting API
api = FastAPI(
    title='Sentiment analysis',
    description="Petite API de projet Sentiement analysis de Datascientest",
    version="1.0.1"
)

#######################################################################################################################
#BASIC AUTH
#######################################################################################################################
user_database = {
  "datascientest": "datascientest",
  }

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def authenticate_user(username, password):
  if password == user_database[username]:
    return True
  else : 
    return False

#Authentication
@api.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username
    password = form_data.password

    if authenticate_user(username, password):
        return {"access_token": username, "token_type": "bearer"}
    else :
        raise HTTPException(status_code=400, detail="Incorrect username or password")

#######################################################################################################################
#APP
#######################################################################################################################

#importing joblib model
model = joblib.load('LR_pipe.joblib')

@api.get("/", name="Home page de l'API")
def home(token: str = Depends(oauth2_scheme)):
    """Root
    """
    return {"token": token}
 #   return {'data': 'Bonjour, bienvenue sur l"API'}

@api.get('/status', name="Getting status")
async def get_status():
    """Returns "OK service online" if status online
    """
    return "OK service online"

#Main endpoint
def transform_data(df):
    return df.to_json(orient="records")
@api.post('/result_with_csv', name='Post data csv_file and get result as a dataframe in json format')
async def upload_data(csv_file: UploadFile = File(...), token: str = Depends(oauth2_scheme)):
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

@api.post('/result_with_text/{user_text}', name='Post data text and get result as text')
async def upload_data_text(user_text, token: str = Depends(oauth2_scheme)):
    """Load data and return a json file of the dataset with the result.
    """
    X = text_transform(user_text)
    result = model.predict(X)
    if result == 1:
        return {"Positive comment"}
    else :
        return {"Negative comment"}
