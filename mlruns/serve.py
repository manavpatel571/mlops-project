from fastapi import FastAPI
import mlflow
import pandas as pd
import os

app = FastAPI()

# Load model from local mlruns folder (inside Docker)
model_uri = os.path.join(os.getcwd(), "mlruns/826412763467474429/models/m-411b9a0aa94b4250bae33fa668a52601/artifacts")
model = mlflow.pyfunc.load_model(model_uri)

@app.get("/")
def home():
    return {"message": "MLflow model is live inside Docker!"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}
