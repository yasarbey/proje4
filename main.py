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

def load_models():
    global model, scaler, le, yamnet_model
    try:
        print("Modeller yükleniyor...")
        
        # YAMNet modelini önce yükle
        print("YAMNet yükleniyor...")
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        
        # Ana model yükleme
        print("Ana model yükleniyor...")
        model = tf.keras.models.load_model(
            'model/turkce_duygu_modeli_enhanced.h5',
            custom_objects={'time_major': lambda x: x},
            compile=False  # Compile etmeden yükle, daha hızlı
        )
        
        # Diğer dosyaları yükle
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
if not load_models():
    raise Exception("Modeller yüklenemedi!")

def extract_prosodic_features(audio, sr):
   try:
       # F0 contour
       f0, voiced_flag, voiced_probs = librosa.pyin(audio,
                                                   fmin=librosa.note_to_hz('C2'),
                                                   fmax=librosa.note_to_hz('C7'))
       f0_mean = np.mean(f0[np.isfinite(f0)])
       f0_std = np.std(f0[np.isfinite(f0)])

       # Energy
       energy = np.sum(audio ** 2) / len(audio)

       # Speaking rate
       zero_crossings = librosa.zero_crossings(audio)
       speaking_rate = sum(zero_crossings) / (len(audio) / sr)

       return np.array([f0_mean, f0_std, energy, speaking_rate])
   except Exception as e:
       print(f"Prosodic özellik çıkarma hatası: {str(e)}")
       return np.array([0, 0, 0, 0])

def extract_features(audio, sr):
   # YAMNet özellikler
   wav_data = tf.convert_to_tensor(audio, dtype=tf.float32)
   _, embeddings, _ = yamnet_model(wav_data)
   yamnet_features = tf.reduce_mean(embeddings, axis=0)

   # MFCC
   mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
   mfccs_mean = np.mean(mfccs.T, axis=0)
   mfccs_var = np.var(mfccs.T, axis=0)

   # Chroma
   chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
   chroma_mean = np.mean(chroma.T, axis=0)

   # Mel spektrogramı
   mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
   mel_spec_mean = np.mean(mel_spec.T, axis=0)

   # Spektral kontrast
   contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
   contrast_mean = np.mean(contrast.T, axis=0)

   # Prosodik özellikler
   prosodic_features = extract_prosodic_features(audio, sr)

   features = np.concatenate([
       yamnet_features.numpy(),
       mfccs_mean,
       mfccs_var,
       chroma_mean,
       mel_spec_mean,
       contrast_mean,
       prosodic_features
   ])

   return np.nan_to_num(features)

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

       # YAMNet özellik çıkarma
       print("[6] YAMNet özellikleri çıkarılıyor...")
       wav_data = tf.convert_to_tensor(audio, dtype=tf.float32)
       _, embeddings, _ = yamnet_model(wav_data)
       yamnet_features = tf.reduce_mean(embeddings, axis=0)
       print("[7] YAMNet özellikleri çıkarıldı")

       # MFCC özellikleri
       print("[8] MFCC özellikleri hesaplanıyor...")
       mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
       mfccs_mean = np.mean(mfccs.T, axis=0)
       mfccs_var = np.var(mfccs.T, axis=0)
       print("[9] MFCC özellikleri hesaplandı")

       # Chroma özellikleri
       print("[10] Chroma özellikleri hesaplanıyor...")
       chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
       chroma_mean = np.mean(chroma.T, axis=0)
       print("[11] Chroma özellikleri hesaplandı")

       # Mel spektrogramı
       print("[12] Mel spektrogramı hesaplanıyor...")
       mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
       mel_spec_mean = np.mean(mel_spec.T, axis=0)
       print("[13] Mel spektrogramı hesaplandı")

       # Spektral kontrast
       print("[14] Spektral kontrast hesaplanıyor...")
       contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
       contrast_mean = np.mean(contrast.T, axis=0)
       print("[15] Spektral kontrast hesaplandı")

       # Prosodik özellikler
       print("[16] Prosodik özellikler hesaplanıyor...")
       prosodic_features = extract_prosodic_features(audio, sr)
       print("[17] Prosodik özellikler hesaplandı")

       # Özellikleri birleştir
       print("[18] Özellikler birleştiriliyor...")
       features = np.concatenate([
           yamnet_features.numpy(),
           mfccs_mean,
           mfccs_var,
           chroma_mean,
           mel_spec_mean,
           contrast_mean,
           prosodic_features
       ])
       print("[19] Özellikler birleştirildi")

       # Özellikleri ölçekle
       print("[20] Özellikler ölçekleniyor...")
       scaled_features = scaler.transform([features])[0]
       print("[21] Özellikler ölçeklendi")

       # Tahmin yap
       print("[22] Model tahmini yapılıyor...")
       prediction = model.predict(np.array([scaled_features]))
       print("[23] Model tahmini tamamlandı")

       # Tahmin sonuçları
       emotion = le.classes_[np.argmax(prediction)]
       confidence = float(np.max(prediction))
       print(f"[24] Tahmin tamamlandı: {emotion}, Güven: {confidence}")

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
        workers=1,  # Railway'de tek worker kullanmak daha güvenli
        loop="auto"
    )
