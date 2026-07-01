import mlflow
import pandas as pd
from fastapi import FastAPI, Request
from prometheus_client import make_asgi_app, Counter

app = FastAPI()

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

print("Memuat model asli dari direktori lokal...")
model = mlflow.pyfunc.load_model("model_lokal")

@app.post("/invocations")
async def predict(request: Request):
    REQUEST_COUNT.inc() 
    
    data = await request.json()
    
    df = pd.DataFrame(
        data['dataframe_split']['data'], 
        columns=data['dataframe_split']['columns']
    )
    
    predictions = model.predict(df)
    
    return {"predictions": predictions.tolist()}