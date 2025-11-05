from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc

app = FastAPI(title="MLflow FastAPI Model Server")

MODEL_URI = "mlruns/826412763467474429/models/m-411b9a0aa94b4250bae33fa668a52601/artifacts"

try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print("✅ MLflow model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load MLflow model: {e}")
    model = None

@app.get("/")
def home():
    if model is not None:
        return {"message": "MLflow model is live inside Docker!"}
    else:
        return {"message": "MLflow model not loaded!"}

# Dynamically get input features
MODEL_FEATURES = []
if model:
    try:
        MODEL_FEATURES = list(model.metadata.get_input_schema().input_names)
    except Exception:
        MODEL_FEATURES = ["feature1", "feature2", "feature3"]

InputDataDynamic = type(
    "InputDataDynamic",
    (BaseModel,),
    {feat: (float, ...) for feat in MODEL_FEATURES}
)

@app.post("/predict")
def predict(data: InputDataDynamic):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    input_df = pd.DataFrame([data.dict()])
    print("Input DataFrame:", input_df)

    try:
        prediction = model.predict(input_df)
        if hasattr(prediction, "tolist"):
            prediction = prediction.tolist()
        elif isinstance(prediction, pd.DataFrame):
            prediction = prediction.to_dict(orient="records")
        print("Prediction output:", prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {"prediction": prediction}
