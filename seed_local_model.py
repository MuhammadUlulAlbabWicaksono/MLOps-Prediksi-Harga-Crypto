import os
import mlflow
import mlflow.pyfunc

# Mengarahkan SDK agar mengisi MLflow lokal yang ada di Docker
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadminSecretKey"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

class DummyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return [65000.0] * len(model_input)

mlflow.set_experiment("Local_Test")
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="xgboost-model",
        python_model=DummyModel(),
        registered_model_name="BTC_Price_Predictor"
    )

client = mlflow.tracking.MlflowClient()
model_version = client.get_latest_versions("BTC_Price_Predictor", stages=["None"])[0].version
client.set_registered_model_alias("BTC_Price_Predictor", "staging", model_version)
print("Dummy model berhasil ditanam ke MLflow & MinIO Lokal!")
