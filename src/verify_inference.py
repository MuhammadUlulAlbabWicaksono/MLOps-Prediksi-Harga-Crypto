import dagshub
import mlflow.pyfunc
import pandas as pd

# Inisialisasi koneksi bersih
dagshub.init(repo_owner='MuhammadUlulAlbabWicaksono', repo_name='MLOps-Prediksi-Harga-Crypto', mlflow=True)

def verify():
    # Perhatikan: menggunakan '@production' (huruf p kecil) sesuai sistem MLflow
    model_uri = "models:/BTC_Price_Predictor@production"
    print(f"Mengunduh model dari: {model_uri} ...")
    
    try:
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        print("Model berhasil dimuat!\n")
        
        # Data tiruan
        dummy_data = pd.DataFrame([{
            "price": 64000.50,
            "market_cap": 1200000000000,
            "volume": 35000000000,
            "MA_7": 63500.10,
            "MA_30": 61200.45,
            "RSI_14": 55.5,
            "MACD": 120.4,
            "MACD_Signal": 110.2
        }])
        
        prediksi = model.predict(dummy_data)
        
        print("=== VERIFIKASI BERHASIL ===")
        print(f"Estimasi Harga Target: {prediksi[0]:.2f}")
        
    except Exception as e:
        print(f"\n=== VERIFIKASI GAGAL ===")
        print(f"Error: {e}")

if __name__ == "__main__":
    verify()