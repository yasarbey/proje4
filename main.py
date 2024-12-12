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
import logging

# Logging ayarları
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        logger.info("Modeller yükleniyor...")
        
        # YAMNet modelini önce yükle
        logger.info("YAMNet yükleniyor...")
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        
        # Ana model yükleme
        logger.info("Ana model yükleniyor...")
        model = tf.keras.models.load_model(
            'model/turkce_duygu_modeli_enhanced.h5',
            custom_objects={'time_major': lambda x: x},
            compile=False  # Compile etmeden yükle, daha hızlı
        )
        
        # Diğer dosyaları yükle
        logger.info("Yardımcı modeller yükleniyor...")
        scaler = joblib.load('model/scaler.pkl')
        with open('model/label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
            
        logger.info("Tüm modeller başarıyla yüklendi")
        return True
    except Exception as e:
        logger.error(f"Model yükleme hatası: {str(e)}")
        return False

# Uygulama başlatılırken modelleri yükle
logger.info("Uygulama başlatılıyor...")
if not load_models():
    raise Exception("Modeller yüklenemedi!")

def extract_prosodic_features(audio, sr):
   try:
       # F0 contour - daha optimize edilmiş
       f0, voiced_flag, voiced_probs = librosa.pyin(
           audio,
           fmin=librosa.note_to_hz('C2'),
           fmax=librosa.note_to_hz('C7'),
           frame_length=2048,  # Daha büyük frame size
           hop_length=512      # Daha büyük hop length
       )
       f0_mean = np.mean(f0[np.isfinite(f0)])
       f0_std = np.std(f0[np.isfinite(f0)])

       # Energy
       energy = np.sum(audio ** 2) / len(audio)

       # Speaking rate
       zero_crossings = librosa.zero_crossings(audio, pad=False)
       speaking_rate = sum(zero_crossings) / (len(audio) / sr)

       return np.array([f0_mean, f0_std, energy, speaking_rate])
   except Exception as e:
       logger.error(f"Prosodic özellik çıkarma hatası: {str(e)}")
       return np.array([0, 0, 0, 0])

def extract_features(audio, sr):
    try:
        # Ses verisini normalize et
        audio = audio / np.max(np.abs(audio))
        
        # YAMNet özellikler
        wav_data = tf.convert_to_tensor(audio, dtype=tf.float32)
        _, embeddings, _ = yamnet_model(wav_data)
        yamnet_features = tf.reduce_mean(embeddings, axis=0)

        # MFCC ve diğer özellikleri daha verimli hesapla
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=20,
            hop_length=512,  # Daha büyük hop length
            n_fft=2048      # Daha büyük FFT window
        )
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_var = np.var(mfccs.T, axis=0)

        # Diğer özellikler
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=512)
        chroma_mean = np.mean(chroma.T, axis=0)

        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=512)
        mel_spec_mean = np.mean(mel_spec.T, axis=0)

        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=512)
        contrast_mean = np.mean(contrast.T, axis=0)

        # Prosodik özellikler
        prosodic_features = extract_prosodic_features(audio, sr)

        # Tüm özellikleri birleştir
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
    except Exception as e:
        logger.error(f"Feature extraction hatası: {str(e)}")
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
       logger.info(f"Dosya alındı: {file.filename}")

       # Dosya kontrolü
       if not file.filename.endswith('.wav'):
           raise HTTPException(status_code=400, detail="Sadece .wav formatında dosyalar kabul edilir.")

       # Dosyayı oku
       contents = await file.read()
       if not contents:
           raise HTTPException(status_code=400, detail="Gönderilen dosya boş.")

       logger.info(f"Dosya boyutu: {len(contents)} bytes")

       # Ses dosyasını librosa ile yükle
       audio, sr = librosa.load(BytesIO(contents), sr=None, duration=30)  # Max 30 saniyelik ses
       logger.info(f"Ses yüklendi - örnek sayısı: {len(audio)}, örnek oranı: {sr}")

       # Özellik çıkarma
       features = extract_features(audio, sr)
       logger.info("Özellikler çıkarıldı")

       # Özellikleri ölçekle
       scaled_features = scaler.transform([features])[0]
       prediction = model.predict(np.array([scaled_features]), verbose=0)

       # Tahmin sonuçları
       emotion = le.classes_[np.argmax(prediction)]
       confidence = float(np.max(prediction))
       logger.info(f"Tahmin: {emotion}, Güven: {confidence}")

       return {
           "status": "success",
           "emotion": emotion,
           "confidence": confidence
       }

   except Exception as e:
       logger.error(f"Hata: {str(e)}")
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
