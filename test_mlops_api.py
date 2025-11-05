import requests

BASE_URL = "http://localhost:8000"

def generate_sample_input(features):
    return {feat: 1.0 for feat in features}

def check_health():
    try:
        response = requests.get(f"{BASE_URL}/")
        print("Health Check Response:", response.json())
    except Exception as e:
        print("Error connecting to FastAPI:", e)

def get_features():
    return ["feature1", "feature2", "feature3"]  # fallback features if schema unavailable

def make_prediction():
    features = get_features()
    data = generate_sample_input(features)

    try:
        response = requests.post(f"{BASE_URL}/predict", json=data)
        print("Prediction Response:", response.json())
    except Exception as e:
        print("Error making prediction:", e)

if __name__ == "__main__":
    check_health()
    make_prediction()
