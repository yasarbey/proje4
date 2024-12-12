from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import librosa
import joblib
import pickle
import os
import tensorflow_hub as hub
from io import BytesIO

app = FastAPI(
   title="Duygu Analizi API",
   description="Ses kayıtlarından duygu analizi yapan API",
   version="1.0.0"
)

# CORS ayarları
app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)

# Global değişkenler
model = None
scaler = None
le = None
yamnet_model = None

# YAMNet modelini yükle
def initialize_yamnet():
    global yamnet_model
    try:
        print("YAMNet modeli yükleniyor...")
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        print("YAMNet modeli başarıyla yüklendi")
        return True
    except Exception as e:
        print(f"YAMNet yükleme hatası: {str(e)}")
        return False

# Diğer modelleri yükle
def initialize_models():
    global model, scaler, le
    try:
        print("Ana model yükleniyor...")
        model = tf.keras.models.load_model(
            'model/turkce_duygu_modeli_enhanced.h5',
            custom_objects={'time_major': lambda x: x},
            compile=False
        )
        
        print("Yardımcı modeller yükleniyor...")
        scaler = joblib.load('model/scaler.pkl')
        with open('model/label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
            
        print("Tüm modeller başarıyla yüklendi")
        return True
    except Exception as e:
        print(f"Model yükleme hatası: {str(e)}")
        return False

# Uygulama başlatılırken modelleri yükle
print("Uygulama başlatılıyor...")
if not initialize_yamnet():
    raise Exception("YAMNet modeli yüklenemedi!")
if not initialize_models():
    raise Exception("Modeller yüklenemedi!")

def extract_features(audio, sr):
    try:
        print("[6] YAMNet özellikleri çıkarılıyor...")
        wav_data = tf.convert_to_tensor(audio, dtype=tf.float32)
        _, embeddings, _ = yamnet_model(wav_data)
        yamnet_features = tf.reduce_mean(embeddings, axis=0)
        print("[7] YAMNet özellikleri çıkarıldı")

        print("[8] MFCC özellikleri hesaplanıyor...")
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        mfcc_features = np.concatenate([
            np.mean(mfccs.T, axis=0),
            np.var(mfccs.T, axis=0)
        ])
        print("[9] MFCC özellikleri hesaplandı")

        features = np.concatenate([
            yamnet_features.numpy(),
            mfcc_features
        ])
        return np.nan_to_num(features)
    except Exception as e:
        print(f"Feature extraction hatası: {str(e)}")
        raise e

# Root endpoints
@app.get("/")
def read_root():
   return {"message": "API çalışıyor",
           "endpoints": {
               "POST /": "Ses analizi için kullanın",
               "GET /": "API durumunu kontrol edin"
           }}

@app.post("/")
async def analyze_emotion(file: UploadFile = File(...)):
   try:
       print(f"[1] Dosya alındı: {file.filename}")

       # Dosya kontrolü
       if not file.filename.endswith('.wav'):
           raise HTTPException(status_code=400, detail="Sadece .wav formatında dosyalar kabul edilir.")

       # Dosyayı oku
       print("[2] Dosya okuma başladı...")
       contents = await file.read()
       if not contents:
           raise HTTPException(status_code=400, detail="Gönderilen dosya boş.")
       print(f"[3] Dosya okundu, boyut: {len(contents)} bytes")

       # Ses dosyasını librosa ile yükle
       print("[4] Librosa ile ses yükleme başladı...")
       audio, sr = librosa.load(BytesIO(contents), sr=None)
       print(f"[5] Ses yüklendi - örnek sayısı: {len(audio)}, örnek oranı: {sr}")

       # Özellik çıkarma
       features = extract_features(audio, sr)
       print("[10] Tüm özellikler çıkarıldı")

       # Özellikleri ölçekle
       print("[11] Özellikler ölçekleniyor...")
       scaled_features = scaler.transform([features])[0]
       print("[12] Özellikler ölçeklendi")

       # Tahmin yap
       print("[13] Model tahmini yapılıyor...")
       prediction = model.predict(np.array([scaled_features]))
       print("[14] Model tahmini tamamlandı")

       # Tahmin sonuçları
       emotion = le.classes_[np.argmax(prediction)]
       confidence = float(np.max(prediction))
       print(f"[15] Tahmin tamamlandı: {emotion}, Güven: {confidence}")

       return {
           "status": "success",
           "emotion": emotion,
           "confidence": confidence
       }

   except Exception as e:
       print(f"HATA: {str(e)}")
       print(f"Hata konumu: {e.__traceback__.tb_frame.f_code.co_name}")
       print(f"Satır numarası: {e.__traceback__.tb_lineno}")
       raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        timeout_keep_alive=600,
        workers=1,
        loop="auto"
    )
