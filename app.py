import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from schema import Customer
from fastapi import HTTPException
import traceback


app = FastAPI(
    title = "Bank Marketing Campaign API",
    description =  "Predict the bank term deposit would be yes or no subscribed from preprocessing and SVC",
    version = '1.0.0'
)


model = joblib.load("model/Bank_Marketing_pipeline.joblib")     

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return{"status":"okk"}


@app.post("/predict")
def predict(customer:Customer):
        data = pd.DataFrame([customer.model_dump()])
        prediction = int(model.predict(data)[0])
        label = "YES" if prediction == 1 else "NO"
        return{
              "prediction": prediction,
              "label": label
        }



        

    
     
