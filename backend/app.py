from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import os
import json
import logging
import warnings
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore, auth
import io
import math
import requests
import time
import gc

# Try importing cv2 for image processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

warnings.filterwarnings('ignore')

# =======================
# APP SETUP
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # This is 'backend' folder
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # Go up one level to project root

# Frontend directory
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# Check if frontend directory exists
if not os.path.exists(FRONTEND_DIR):
    print(f"⚠️ WARNING: Frontend directory not found at: {FRONTEND_DIR}")
    print("Creating frontend directory...")
    os.makedirs(FRONTEND_DIR, exist_ok=True)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__,
            static_folder=FRONTEND_DIR,
            static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Configure CORS with explicit domains
ALLOWED_ORIGINS = [
    "http://localhost:5000",
    "http://localhost:8080",
    "http://127.0.0.1:5000",
    "http://127.0.0.1:8080",
    "https://voiceguard-5db49.web.app",
    "https://voiceguard-5db49.firebaseapp.com",
    "https://voiceguard-backend-new.onrender.com"
]

CORS(app, resources={
    r"/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
        "supports_credentials": True
    }
})



@app.errorhandler(500)
def handle_500_error(e):
    logger.error(f"Internal Server Error: {str(e)}")
    return jsonify({
        "success": False,
        "error": "Internal Server Error - The server might be restarting or out of memory.",
        "details": str(e)
    }), 500

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =======================
# FIREBASE ADMIN SETUP
# =======================
try:
    # Try to get Firebase credentials from environment variable
    firebase_creds_json = os.environ.get('FIREBASE_CREDENTIALS')
    
    if firebase_creds_json:
        # Parse JSON string from environment variable
        try:
            cred_dict = json.loads(firebase_creds_json)
            # If cred_dict is still a string (double-encoded), load it again
            if isinstance(cred_dict, str):
                logger.info("💡 Firebase credentials appear to be double-encoded, parsing again...")
                cred_dict = json.loads(cred_dict)
            
            cred = credentials.Certificate(cred_dict)
            logger.info("✅ Firebase credentials loaded from environment variable")
        except Exception as parse_err:
            logger.error(f"❌ Failed to parse FIREBASE_CREDENTIALS JSON: {str(parse_err)}")
            # Fall through to other loading methods
            cred = None
    
    if not firebase_creds_json or not cred:
        # Try to load from file
        cred_path = os.path.join(BASE_DIR, 'firebase-credentials.json')
        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
        else:
            # Create dummy credentials for development
            print("⚠️ Firebase credentials not found. Using dummy credentials for development.")
            cred = credentials.Certificate({
                "type": "service_account",
                "project_id": "voiceguard-5db49",
                "private_key_id": "dummy",
                "private_key": "-----BEGIN PRIVATE KEY-----\ndummy\n-----END PRIVATE KEY-----\n",
                "client_email": "dummy@voiceguard-5db49.iam.gserviceaccount.com",
                "client_id": "dummy",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/dummy%40voiceguard-5db49.iam.gserviceaccount.com"
            })
    
    # Initialize Firebase Admin
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase Admin initialized successfully!")
    
except Exception as e:
    print(f"❌ Firebase Admin initialization failed: {str(e)}")
    print("⚠️ Running without Firebase Admin features")
    db = None

# =======================
# PRINT DIRECTORY INFO
# =======================
print("\n" + "=" * 70)
print("🎤 VOICEGUARD AI - DIRECTORY CONFIGURATION")
print("=" * 70)
print(f"📁 Backend Directory: {BASE_DIR}")
print(f"📁 Project Root: {PROJECT_ROOT}")
print(f"📁 Frontend Directory: {FRONTEND_DIR}")
print(f"📁 Models Directory: {MODEL_DIR}")
print(f"🔥 Firebase Available: {db is not None}")

# Check if index.html exists
index_path = os.path.join(FRONTEND_DIR, "index.html")
if os.path.exists(index_path):
    print(f"✅ Frontend found: {index_path}")
else:
    print(f"❌ Frontend not found at: {index_path}")
    print("Please place index.html in the 'frontend' folder")

print("=" * 70)

# =======================
# MEMORY MONITORING UTILITY
# =======================
def get_memory_usage():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / (1024 * 1024)
        return mem
    except:
        return 0

def log_memory(stage=""):
    mem = get_memory_usage()
    if stage:
        logger.info(f"💾 Memory Usage [{stage}]: {mem:.2f} MB")
    else:
        logger.info(f"💾 Memory Usage: {mem:.2f} MB")

# =======================
# MODELS (GLOBAL STATE)
# =======================
covid_model = None
covid_model_is_tflite = False
disease_model = None
disease_scaler = None
model_classes = ["Normal_Cold", "COVID", "Asthma", "Bronchitis"]

# Flag definitions (will be updated by load functions)
NUMPY_AVAILABLE = False
LIBROSA_AVAILABLE = False
CV2_AVAILABLE = False # This is for ML-related CV2, distinct from CV2_AVAILABLE_INITIAL
TFLITE_RUNTIME_AVAILABLE = False
TENSORFLOW_AVAILABLE = False
JOBLIB_AVAILABLE = False
SCIPY_AVAILABLE = False

def _import_numpy():
    global NUMPY_AVAILABLE
    try:
        import numpy as np
        NUMPY_AVAILABLE = True
        logger.info("✅ NumPy imported successfully")
        return np
    except ImportError as e:
        logger.error(f"❌ NumPy not available: {str(e)}")
        NUMPY_AVAILABLE = False
        return None

def _import_librosa():
    global LIBROSA_AVAILABLE
    try:
        import librosa
        LIBROSA_AVAILABLE = True
        logger.info("✅ Librosa imported successfully")
        return librosa
    except ImportError as e:
        logger.error(f"❌ Librosa not available: {str(e)}")
        LIBROSA_AVAILABLE = False
        return None

def _import_cv2():
    global CV2_AVAILABLE
    try:
        import cv2
        CV2_AVAILABLE = True
        logger.info("✅ OpenCV imported successfully")
        return cv2
    except ImportError as e:
        logger.error(f"❌ OpenCV not available: {str(e)}")
        CV2_AVAILABLE = False
        return None

def _import_tflite_tensorflow():
    global TFLITE_RUNTIME_AVAILABLE, TENSORFLOW_AVAILABLE
    tflite_module = None
    tf_module = None
    try:
        import tflite_runtime.interpreter as tflite_runtime_interpreter
        TFLITE_RUNTIME_AVAILABLE = True
        logger.info("✅ tflite-runtime imported successfully")
        tflite_module = tflite_runtime_interpreter
    except ImportError:
        TFLITE_RUNTIME_AVAILABLE = False
        try:
            import tensorflow as tf
            TENSORFLOW_AVAILABLE = True
            logger.info("✅ TensorFlow imported successfully (fallback)")
            tf_module = tf
        except ImportError as e:
            logger.error(f"❌ Neither tflite-runtime nor TensorFlow available: {str(e)}")
            TENSORFLOW_AVAILABLE = False
    return tflite_module, tf_module

def _import_joblib():
    global JOBLIB_AVAILABLE
    try:
        import joblib
        JOBLIB_AVAILABLE = True
        logger.info("✅ Joblib imported successfully")
        return joblib
    except ImportError as e:
        logger.error(f"❌ Joblib not available: {str(e)}")
        JOBLIB_AVAILABLE = False
        return None

def _import_soundfile():
    try:
        import soundfile as sf
        logger.info("✅ Soundfile imported successfully")
        return sf
    except ImportError as e:
        logger.error(f"❌ Soundfile not available: {str(e)}")
        return None

def _import_soxr():
    try:
        import soxr
        logger.info("✅ soxr imported successfully")
        return soxr
    except ImportError as e:
        logger.error(f"❌ soxr not available: {str(e)}")
        return None

def _import_scipy_signal():
    global SCIPY_AVAILABLE
    try:
        from scipy import signal
        SCIPY_AVAILABLE = True
        logger.info("✅ SciPy imported successfully")
        return signal
    except ImportError as e:
        logger.error(f"❌ SciPy not available: {str(e)}")
        SCIPY_AVAILABLE = False
        return None

# =======================
# MODELS
# =======================
covid_model = None
covid_model_is_tflite = False
disease_model = None
disease_scaler = None
model_classes = ["Normal_Cold", "COVID", "Asthma", "Bronchitis"]  # Default order

def load_or_create_models():
    """Load existing models"""
    global covid_model, covid_model_is_tflite, disease_model, disease_scaler, model_classes
    
    log_memory("Startup - Before loading models")
    
    # Try to load COVID model
    disable_heavy_ml = os.environ.get('RENDER_LITE_MODE', 'false').lower() == 'true'
    
    # 1. COVID MODEL LOADING (TFLite or Keras)
    if not disable_heavy_ml:
        tflite_path = os.path.join(MODEL_DIR, "voiceguard_audio_model_final.tflite")
        if os.path.exists(tflite_path):
            try:
                # Lazy import for TFLite
                tflite_module, tf_module = _import_tflite_tensorflow()
                
                if tflite_module or tf_module:
                    logger.info(f"⚡ Loading TFLite model from: {tflite_path}")
                    if tflite_module:
                        interpreter = tflite_module.Interpreter(model_path=tflite_path)
                    else:
                        interpreter = tf_module.lite.Interpreter(model_path=tflite_path)
                        
                    interpreter.allocate_tensors()
                    covid_model = interpreter
                    covid_model_is_tflite = True
                    logger.info("✅ COVID TFLite model loaded successfully!")
                    log_memory("After TFLite load")
            except Exception as tflite_err:
                logger.error(f"❌ Failed to load TFLite model: {str(tflite_err)}")
        
        # 2. Try Keras as Fallback (Only if TFLite failed and full TF is available)
        if not covid_model:
            covid_model_path = os.path.join(MODEL_DIR, "voiceguard_audio_model_final.keras")
            if os.path.exists(covid_model_path):
                try:
                    _, tf_module = _import_tflite_tensorflow()
                    if tf_module:
                        logger.info(f"Attempting to load COVID Keras model from: {covid_model_path}")
                        covid_model = tf_module.keras.models.load_model(covid_model_path, compile=False)
                        covid_model_is_tflite = False
                        logger.info("✅ COVID audio model loaded successfully via tf.keras!")
                        log_memory("After Keras load")
                except Exception as keras_err:
                    logger.error(f"❌ Failed to load Keras model: {str(keras_err)}")
    else:
        logger.info("🔌 Lite Mode: Skipping COVID audio model loading to save memory")
    
    # 3. DISEASE CLASSIFICATION MODEL LOADING (Pickle)
    disease_model_path = os.path.join(MODEL_DIR, "disease_classification_model.pkl")
    scaler_paths = [
        os.path.join(MODEL_DIR, "disease_scaler.pkl"),
        os.path.join(MODEL_DIR, "disease_scaler (1).pkl")
    ]
    
    if os.path.exists(disease_model_path):
        try:
            joblib_module = _import_joblib()
            if joblib_module:
                disease_model = joblib_module.load(disease_model_path)
                logger.info(f"✅ Disease classification model loaded from: {disease_model_path}")
                
                # Try to get classes from model
                if hasattr(disease_model, 'classes_'):
                    model_classes = disease_model.classes_
                    logger.info(f"   Model classes: {model_classes}")
                
                # Load scaler (trying multiple paths)
                for scaler_path in scaler_paths:
                    if os.path.exists(scaler_path):
                        disease_scaler = joblib_module.load(scaler_path)
                        logger.info(f"✅ Disease scaler loaded from: {os.path.basename(scaler_path)}")
                        break
                
                log_memory("After Disease models load")
        except Exception as e:
            logger.error(f"❌ Failed to load disease classification model: {str(e)}")
    
    log_memory("End of load_or_create_models")

# Load models
load_or_create_models()

# =======================
# FEATURE ENGINEERING (23 FEATURES)
# =======================
def enhance_features(symptoms_dict, age, cough_days):
    """
    Create 23 engineered features from basic symptoms
    """
    # Lazy import for numpy
    np_module = _import_numpy()
    if not np_module:
        logger.error("❌ NumPy not available for feature engineering")
        return None
    
    try:
        # Convert symptoms to binary
        fever = 1 if symptoms_dict.get('fever', 'no') == 'yes' else 0
        dry_cough = 1 if symptoms_dict.get('dry_cough', 'no') == 'yes' else 0
        wet_cough = 1 if symptoms_dict.get('wet_cough', 'no') == 'yes' else 0
        wheezing = 1 if symptoms_dict.get('wheezing', 'no') == 'yes' else 0
        breath = 1 if symptoms_dict.get('breath', 'no') == 'yes' else 0
        chest = 1 if symptoms_dict.get('chest', 'no') == 'yes' else 0
        fatigue = 1 if symptoms_dict.get('fatigue', 'no') == 'yes' else 0
        sore_throat = 1 if symptoms_dict.get('sore_throat', 'no') == 'yes' else 0
        smell = 1 if symptoms_dict.get('smell', 'no') == 'yes' else 0
        night_cough = 1 if symptoms_dict.get('night_cough', 'no') == 'yes' else 0
        is_smoker = 1 if symptoms_dict.get('smoker', 'no') == 'yes' else 0
        is_chronic = 1 if symptoms_dict.get('chronic', 'no') == 'yes' else 0
        
        # Pattern features
        asthma_pattern = 1 if (wheezing == 1 and night_cough == 1 and fever == 0) else 0
        covid_pattern = 1 if (smell == 1 or (fever == 1 and dry_cough == 1 and fatigue == 1)) else 0
        bronchitis_pattern = 1 if (wet_cough == 1 and is_smoker == 1 and cough_days > 14) else 0
        
        # Composite scores
        respiratory_symptoms = [wheezing, breath, chest, dry_cough, wet_cough]
        systemic_symptoms = [fever, fatigue]
        distinctive_symptoms = [smell, night_cough]
        
        respiratory_score = sum(respiratory_symptoms)
        systemic_score = sum(systemic_symptoms)
        distinctive_score = sum(distinctive_symptoms)
        total_symptoms = respiratory_score + systemic_score + distinctive_score
        
        # Age grouping
        if age <= 30:
            age_group = 0
        elif age <= 50:
            age_group = 1
        elif age <= 70:
            age_group = 2
        else:
            age_group = 3
        
        # Duration categorization
        if cough_days <= 7:
            duration_category = 0
        elif cough_days <= 14:
            duration_category = 1
        elif cough_days <= 30:
            duration_category = 2
        else:
            duration_category = 3
        
        features = [
            float(age),              # 0: Age
            float(cough_days),       # 1: Cough_Duration_Days
            float(is_chronic),       # 2: Is_Chronic
            float(is_smoker),        # 3: Is_Smoker
            float(fever),            # 4: Fever
            float(dry_cough),        # 5: Dry Cough
            float(wet_cough),        # 6: Wet Cough
            float(wheezing),         # 7: Wheezing
            float(breath),           # 8: Shortness of Breath
            float(chest),            # 9: Chest Tightness
            float(fatigue),          # 10: Fatigue
            float(sore_throat),      # 11: Sore Throat
            float(smell),            # 12: Loss of Smell/Taste
            float(night_cough),      # 13: Night-time Cough
            float(asthma_pattern),   # 14: Asthma_Pattern
            float(covid_pattern),    # 15: COVID_Pattern
            float(bronchitis_pattern), # 16: Bronchitis_Pattern
            float(respiratory_score), # 17: Respiratory_Score
            float(systemic_score),   # 18: Systemic_Score
            float(distinctive_score), # 19: Distinctive_Score
            float(total_symptoms),   # 20: Total_Symptoms
            float(age_group),        # 21: Age_Group
            float(duration_category) # 22: Duration_Category
        ]
        
        logger.info(f"📊 Engineered features created: {len(features)} features")
        return np_module.array(features).reshape(1, -1)
        
    except Exception as e:
        logger.error(f"❌ Feature engineering failed: {str(e)}")
        return None

# =======================
# CORRECT AUDIO PROCESSING FOR COVID MODEL
# Matches your training code exactly
# =======================
# =======================
# SAFE AUDIO PROCESSING
# =======================
def safe_process_audio(audio_file):
    """
    ULTRA-FAST audio processing:
    1. Check sample rate via sf.info (no load).
    2. Load ONLY 10 seconds from disk (sf.read frames).
    3. Resample directly via soxr (bypassing librosa overhead).
    """
    log_memory("Before safe_process_audio")
    
    # Lazy imports
    sf_module = _import_soundfile()
    soxr_module = _import_soxr()
    librosa_module = _import_librosa()
    np_module = _import_numpy()
    
    if not sf_module or not np_module:
        logger.warning("⚠️ Critical modules (soundfile/numpy) missing, falling back to basic librosa.load")
        if librosa_module:
            try:
                audio_file.seek(0)
                y, sr = librosa_module.load(audio_file, sr=22050, duration=10.0)
                return y, sr
            except:
                return None, "All loaders failed"
        return None, "Required modules for audio loading not available"
    
    try:
        # 1. PEAK DATA (Info Only): Get sample rate without loading samples
        audio_file.seek(0)
        info = sf_module.info(audio_file)
        native_sr = info.samplerate
        
        # 2. DISK-LEVEL SLICE: Read exactly 10s from the file
        max_samples = 10 * native_sr
        audio_file.seek(0)
        logger.info(f"📡 Dish-level load: 10s @ {native_sr}Hz")
        data, _ = sf_module.read(audio_file, frames=max_samples)
        
        # Handle stereo -> mono
        if len(data.shape) > 1:
            data = np_module.mean(data, axis=1)
            
        # 3. FAST RESAMPLE: Use soxr directly for speed
        target_sr = 22050
        if native_sr != target_sr:
            if soxr_module:
                logger.info(f"🔄 Direct soxr resampling: {native_sr}Hz -> {target_sr}Hz")
                y = soxr_module.resample(data, native_sr, target_sr)
            elif librosa_module:
                logger.info(f"🔄 librosa.resample fallback: {native_sr}Hz -> {target_sr}Hz")
                y = librosa_module.resample(y=data, orig_sr=native_sr, target_sr=target_sr)
            else:
                y = data # No resample possible
        else:
            y = data
            
        # Cleanup temp data
        if id(y) != id(data):
            del data
        gc.collect()
        
        log_memory("After Ultra-Fast audio load")
        return y, target_sr
        
    except Exception as e:
        logger.error(f"❌ Ultra-Fast load failed: {str(e)}")
        # Ultimate fallback
        try:
            audio_file.seek(0)
            if librosa_module:
                y, sr = librosa_module.load(audio_file, sr=22050, duration=10.0)
                return y, sr
        except:
            pass
        return None, str(e)

# =======================
def process_audio_for_covid(y, sr):
    """
    Process pre-loaded audio data to match your training data preparation.
    y: numpy array of audio samples
    sr: sample rate
    """
    log_memory("Start process_audio_for_covid")
    
    # Lazy imports
    librosa_module = _import_librosa()
    np_module = _import_numpy()
    cv2_module = _import_cv2()
    
    if not librosa_module or not np_module:
        logger.error("❌ Librosa or NumPy not available for audio processing")
        return None
    
    if not cv2_module:
        logger.error("❌ OpenCV (cv2) not available for image resizing")
        return None
        
    try:
        # Work on a copy to avoid affecting other functions
        y_proc = y.copy()
        
        # Ensure fixed length (3 seconds * 22050 Hz = 66150 samples)
        target_len = 3 * 22050  # 66150 samples
        if len(y_proc) < target_len:
            y_proc = np_module.pad(y_proc, (0, target_len - len(y_proc)), mode='constant')
        else:
            y_proc = y_proc[:target_len]
        
        # Generate mel spectrogram - MATCHES TRAINING PARAMETERS
        mel_spec = librosa_module.feature.melspectrogram(
            y=y_proc, 
            sr=sr, 
            n_mels=128, 
            fmax=8000, 
            n_fft=2048, 
            hop_length=512
        )
        
        # Convert to dB scale (power_to_db)
        mel_spec_db = librosa_module.power_to_db(mel_spec, ref=np_module.max)
        
        # Fixed shape handling - matches training
        if mel_spec_db.shape[1] < 100:
            mel_spec_db = np_module.pad(mel_spec_db, ((0, 0), (0, 100 - mel_spec_db.shape[1])), mode='reflect')
        else:
            mel_spec_db = mel_spec_db[:, :100]
        
        # Resize to 224x224 (model's expected input)
        mel_resized = cv2_module.resize(mel_spec_db, (224, 224), interpolation=cv2_module.INTER_LINEAR)
        
        # Normalization - MATCHES TRAINING
        eps = 1e-8
        mel_normalized = (mel_resized - mel_resized.mean()) / (mel_resized.std() + eps)
        mel_normalized = np_module.clip(mel_normalized, -3, 3)
        mel_normalized = (mel_normalized + 3) / 6  # Scale to [0, 1]
        
        # Stack to 3 channels (RGB)
        mel_rgb = np_module.stack([mel_normalized] * 3, axis=-1)
        
        # Add batch dimension: (1, 224, 224, 3)
        mel_rgb = mel_rgb[np_module.newaxis, ...]
        
        logger.info(f"✅ Audio processed for COVID: shape {mel_rgb.shape}")
        log_memory("End process_audio_for_covid")
        
        # EXPLICIT GARBAGE COLLECTION
        del y_proc
        gc.collect()
        
        return mel_rgb
        
    except Exception as e:
        logger.error(f"❌ Audio processing for COVID failed: {str(e)}")
        return None

# =======================
# AUDIO PREDICTION FUNCTION
# =======================
def predict_from_audio(mel_spectrogram):
    """
    Make prediction using the trained COVID model
    Returns: (prediction_class, confidence, covid_probability)
    Supports both Keras and TFLite
    """
    log_memory("Start predict_from_audio")
    try:
        if covid_model is None:
            logger.warning("⚠️ COVID model not loaded")
            return None, 0.0, 0.0
        
        predictions = None
        
        # Lazy imports for numpy
        np_module = _import_numpy()
        
        # 1. Handle TFLite
        if covid_model_is_tflite:
            input_details = covid_model.get_input_details()
            output_details = covid_model.get_output_details()
            
            # Ensure input is float32
            input_data = mel_spectrogram.astype(np_module.float32)
            
            # Set tensor
            covid_model.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            covid_model.invoke()
            
            # Get result
            predictions = covid_model.get_tensor(output_details[0]['index'])
            
            # Cleanup input data copy
            del input_data
            
        # 2. Handle Keras
        else:
            predictions = covid_model.predict(mel_spectrogram, verbose=0)
        
        log_memory("After model inference")
        
        # Interpret predictions
        if predictions is not None and len(predictions.shape) == 2:
            if predictions.shape[-1] == 2:
                # Binary classification with 2 outputs (softmax)
                covid_probability = float(predictions[0][0])
                normal_probability = float(predictions[0][1])
                
                if covid_probability > normal_probability:
                    prediction = "COVID"
                    confidence = covid_probability
                else:
                    prediction = "Normal_Cold"
                    confidence = normal_probability
                
                logger.info(f"🎤 Audio prediction: COVID prob={covid_probability:.3f}, Normal prob={normal_probability:.3f}")
                return prediction, confidence, covid_probability
                
            elif predictions.shape[-1] == 1:
                covid_probability = float(predictions[0][0])
                if covid_probability > 0.5:
                    prediction = "COVID"
                    confidence = covid_probability
                else:
                    prediction = "Normal_Cold"
                    confidence = 1 - covid_probability
                return prediction, confidence, covid_probability
        
        return None, 0.0, 0.0
        
    except Exception as e:
        logger.error(f"❌ Audio prediction failed: {str(e)}")
        return None, 0.0, 0.0
    finally:
        gc.collect()

# =======================
# SIMPLIFIED AUDIO FEATURES FOR RULES
# (Optional - for rule enhancement)
# =======================
def extract_audio_features_for_rules(y, sr):
    """
    Extract ULTRA-LIGHT audio statistics using only NumPy.
    Bypasses librosa.feature calls which are too slow for 0.1 CPU.
    """
    log_memory("Start extract_audio_features_for_rules")
    
    # Lazy imports
    np_module = _import_numpy()
    
    if not np_module:
        return None
    
    try:
        features = {}
        
        # Slicing is already done in safe_process_audio, but ensure we use 3s for rules
        target_len = 3 * sr
        y_rules = y[:target_len] if len(y) > target_len else y
        
        # 1. RMS (Root Mean Square) - Extremely fast volume detection
        # Helps detect if it's a real recording or silence
        features['rms'] = float(np_module.sqrt(np_module.mean(y_rules**2)))
        
        # 2. Peak Amplitude
        features['peak'] = float(np_module.max(np_module.abs(y_rules)))
        
        # 3. Simple Zero Crossing Estimate (NumPy only)
        # Faster than librosa.feature.zero_crossing_rate
        zero_crossings = np_module.where(np_module.diff(np_module.sign(y_rules)))[0]
        features['zero_crossing_rate'] = float(len(zero_crossings) / len(y_rules))
        
        # Provide dummy values for removed heavy features to avoid breaking downstream logic
        features['spectral_centroid_mean'] = 2000.0 
        features['mel_spectrogram_mean'] = -40.0
        
        logger.info(f"✅ Extracted {len(features)} LIGHT audio features")
        log_memory("End extract_audio_features_for_rules")
        
        return features
        
    except Exception as e:
        logger.error(f"❌ Light audio feature extraction failed: {str(e)}")
        # Return minimal dict to prevent crashes
        return {'rms': 0.0, 'peak': 0.0, 'zero_crossing_rate': 0.0}

# =======================
# RULE-BASED DISEASE DETECTION
# =======================
def rule_based_disease_detection(symptoms_dict, age, cough_days, audio_features=None):
    """
    Comprehensive rule-based disease detection with confidence scoring
    Returns disease, confidence, and detailed reasoning
    """
    try:
        # Extract symptoms
        fever = 1 if symptoms_dict.get('fever', 'no') == 'yes' else 0
        dry_cough = 1 if symptoms_dict.get('dry_cough', 'no') == 'yes' else 0
        wet_cough = 1 if symptoms_dict.get('wet_cough', 'no') == 'yes' else 0
        wheezing = 1 if symptoms_dict.get('wheezing', 'no') == 'yes' else 0
        breath = 1 if symptoms_dict.get('breath', 'no') == 'yes' else 0
        chest = 1 if symptoms_dict.get('chest', 'no') == 'yes' else 0
        fatigue = 1 if symptoms_dict.get('fatigue', 'no') == 'yes' else 0
        sore_throat = 1 if symptoms_dict.get('sore_throat', 'no') == 'yes' else 0
        smell = 1 if symptoms_dict.get('smell', 'no') == 'yes' else 0
        night_cough = 1 if symptoms_dict.get('night_cough', 'no') == 'yes' else 0
        is_smoker = 1 if symptoms_dict.get('smoker', 'no') == 'yes' else 0
        is_chronic = 1 if symptoms_dict.get('chronic', 'no') == 'yes' else 0
        
        # Initialize scores for each disease
        scores = {
            "Normal_Cold": 0,
            "COVID": 0,
            "Asthma": 0,
            "Bronchitis": 0
        }
        
        reasoning = []
        
        # ===== NORMAL COLD RULES =====
        cold_score = 0
        if (fever == 1 or sore_throat == 1) and (dry_cough == 1 or wet_cough == 1):
            cold_score += 30
            reasoning.append("Has fever/sore throat with cough")
        
        if breath == 0 and chest == 0 and wheezing == 0:
            cold_score += 20
            reasoning.append("No severe respiratory symptoms")
        else:
            cold_score -= 15  # Penalize if severe symptoms present
        
        if smell == 0 and fatigue == 0:
            cold_score += 15
            reasoning.append("No COVID-specific symptoms")
        
        if cough_days <= 7:
            cold_score += 15
            reasoning.append("Short duration cough")
        elif cough_days > 14:
            cold_score -= 10  # Long duration less likely for cold
        
        if age < 60:
            cold_score += 10
        else:
            cold_score -= 5
        
        scores["Normal_Cold"] = max(0, min(100, cold_score))
        
        # ===== COVID RULES =====
        covid_score = 0
        
        # Strong indicators
        if smell == 1:
            covid_score += 40
            reasoning.append("Loss of smell/taste - strong COVID indicator")
        
        if fever == 1 and dry_cough == 1 and fatigue == 1:
            covid_score += 30
            reasoning.append("Fever + dry cough + fatigue - classic COVID triad")
        
        if breath == 1:
            covid_score += 20
            reasoning.append("Shortness of breath present")
        
        # Audio features if available
        if audio_features:
            # Use mel spectrogram statistics for COVID detection
            if audio_features.get('mel_spectrogram_mean', 0) < -30:
                # Lower mel values might indicate COVID-specific patterns
                covid_score += 15
                reasoning.append("Audio shows COVID-like spectral patterns")
        
        # Duration factor
        if 5 <= cough_days <= 14:
            covid_score += 10
        elif cough_days > 14:
            covid_score += 5  # Less weight for very long duration
        
        # Age factor
        if age > 50:
            covid_score += 10
        
        scores["COVID"] = max(0, min(100, covid_score))
        
        # ===== ASTHMA RULES =====
        asthma_score = 0
        
        if wheezing == 1:
            asthma_score += 40
            reasoning.append("Wheezing present - key asthma indicator")
        
        if (breath == 1 or chest == 1) and fever == 0:
            asthma_score += 30
            reasoning.append("Breathing difficulty without fever")
        
        if night_cough == 1:
            asthma_score += 20
            reasoning.append("Night-time cough - common in asthma")
        
        if is_chronic == 1:
            asthma_score += 15
            reasoning.append("Has chronic respiratory condition")
        
        if wet_cough == 1:
            asthma_score -= 10  # Wet cough less typical for asthma
        
        if is_smoker == 1:
            asthma_score += 5  # Smokers more prone to asthma
        
        # Age factor
        if age < 40:
            asthma_score += 10
        
        scores["Asthma"] = max(0, min(100, asthma_score))
        
        # ===== BRONCHITIS RULES =====
        bronchitis_score = 0
        
        if wet_cough == 1:
            bronchitis_score += 35
            reasoning.append("Wet cough with mucus")
        
        if cough_days > 14:
            bronchitis_score += 25
            reasoning.append("Cough lasting >2 weeks")
        
        if is_smoker == 1:
            bronchitis_score += 25
            reasoning.append("Smoker - high risk for bronchitis")
        
        if chest == 1:
            bronchitis_score += 15
            reasoning.append("Chest discomfort present")
        
        if is_chronic == 1:
            bronchitis_score += 15
            reasoning.append("History of chronic bronchitis")
        
        if wheezing == 1:
            bronchitis_score += 10  # Can occur in bronchitis
        
        if fever == 1:
            bronchitis_score += 10  # Acute bronchitis may have fever
        
        scores["Bronchitis"] = max(0, min(100, bronchitis_score))
        
        # Find top disease
        top_disease = max(scores, key=scores.get)
        top_score = scores[top_disease]
        
        # Calculate confidence (normalize to 0-1)
        confidence = top_score / 100.0
        
        # Second best for tie-breaking
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        second_score = sorted_scores[1][1]
        
        # If scores are very close, we might have ambiguity
        if top_score - second_score < 10:
            confidence *= 0.9  # Reduce confidence if ambiguous
        
        logger.info(f"📊 Rule-based scores: {scores}")
        logger.info(f"🎯 Top disease: {top_disease} (confidence: {confidence:.2f})")
        
        return top_disease, confidence, scores, reasoning
        
    except Exception as e:
        logger.error(f"❌ Rule-based detection error: {str(e)}")
        return "Normal_Cold", 0.6, {}, []

# =======================
# SEVERITY ASSESSMENT
# =======================
def assess_severity(disease, symptoms_dict, age, cough_days, audio_features=None):
    """
    Comprehensive severity assessment
    Returns severity level (0: Mild, 1: Moderate, 2: Severe) and details
    """
    try:
        # Extract symptoms
        fever = 1 if symptoms_dict.get('fever', 'no') == 'yes' else 0
        breath = 1 if symptoms_dict.get('breath', 'no') == 'yes' else 0
        chest = 1 if symptoms_dict.get('chest', 'no') == 'yes' else 0
        wheezing = 1 if symptoms_dict.get('wheezing', 'no') == 'yes' else 0
        fatigue = 1 if symptoms_dict.get('fatigue', 'no') == 'yes' else 0
        smell = 1 if symptoms_dict.get('smell', 'no') == 'yes' else 0
        
        severity_score = 0
        reasons = []
        
        # 1. Duration-based scoring
        if cough_days > 14:
            severity_score += 30
            reasons.append("Cough lasting >2 weeks")
        elif cough_days > 7:
            severity_score += 15
            reasons.append("Cough lasting >1 week")
        
        # 2. Symptom-based scoring
        if breath == 1:
            severity_score += 25
            reasons.append("Shortness of breath")
        
        if chest == 1:
            severity_score += 20
            reasons.append("Chest tightness/pain")
        
        if wheezing == 1:
            severity_score += 20
            reasons.append("Wheezing")
        
        if fever == 1:
            if age > 65 or age < 5:
                severity_score += 20  # Fever more serious in young/old
                reasons.append("Fever in vulnerable age group")
            else:
                severity_score += 10
                reasons.append("Fever present")
        
        if fatigue == 1:
            severity_score += 5
        
        # 3. Age-based scoring
        if age > 65:
            severity_score += 20
            reasons.append("Age >65 - high risk")
        elif age < 5:
            severity_score += 15
            reasons.append("Age <5 - vulnerable")
        elif age > 50:
            severity_score += 10
            reasons.append("Age >50")
        
        # 4. Disease-specific severity factors
        if disease == "COVID":
            if smell == 1:
                severity_score += 5  # Loss of smell is common but not severe
            if breath == 1:
                severity_score += 15  # Breathing issues in COVID are serious
                reasons.append("COVID with breathing difficulty")
        
        elif disease == "Asthma":
            if wheezing == 1 and breath == 1:
                severity_score += 20
                reasons.append("Active asthma attack symptoms")
        
        elif disease == "Bronchitis":
            is_smoker = 1 if symptoms_dict.get('smoker', 'no') == 'yes' else 0
            if is_smoker == 1 and cough_days > 21:
                severity_score += 15
                reasons.append("Smoker with prolonged bronchitis")
        
        # 5. Audio-based severity (if available)
        if audio_features:
            # Check for wheezing in audio using spectral features
            if 'spectral_centroid_mean' in audio_features:
                if audio_features['spectral_centroid_mean'] < 1500:
                    # Lower spectral centroid might indicate congestion
                    severity_score += 10
                    reasons.append("Audio indicates congestion")
        
        # Determine severity level
        if severity_score >= 60:
            severity_level = 2  # Severe
            severity_text = "Severe"
        elif severity_score >= 30:
            severity_level = 1  # Moderate
            severity_text = "Moderate"
        else:
            severity_level = 0  # Mild
            severity_text = "Mild"
        
        logger.info(f"📊 Severity score: {severity_score} -> {severity_text}")
        
        return severity_level, severity_text, severity_score, reasons
        
    except Exception as e:
        logger.error(f"❌ Severity assessment error: {str(e)}")
        return 0, "Mild", 0, []

# =======================
# NORMAL COLD DETECTION
# =======================
def detect_normal_cold(symptoms_dict, age, cough_days):
    """
    Detect if symptoms indicate a normal cold
    """
    try:
        has_fever = symptoms_dict.get("fever", "no") == "yes"
        has_dry_cough = symptoms_dict.get("dry_cough", "no") == "yes"
        has_sore_throat = symptoms_dict.get("sore_throat", "no") == "yes"
        has_wet_cough = symptoms_dict.get("wet_cough", "no") == "yes"
        
        has_severe_respiratory = (
            symptoms_dict.get("breath", "no") == "yes" or
            symptoms_dict.get("chest", "no") == "yes" or
            symptoms_dict.get("wheezing", "no") == "yes"
        )
        
        has_covid_specific = (
            symptoms_dict.get("smell", "no") == "yes" or
            symptoms_dict.get("fatigue", "no") == "yes"
        )
        
        age = int(age)
        cough_days = int(cough_days)
        
        is_normal_cold = (
            (has_fever or has_sore_throat) and
            (has_dry_cough or has_wet_cough) and
            (not has_severe_respiratory) and
            (not has_covid_specific) and
            (cough_days <= 10) and
            (age < 60)
        )
        
        if cough_days > 14:
            return False
        
        return is_normal_cold
        
    except Exception as e:
        logger.error(f"❌ Normal cold detection error: {str(e)}")
        return False

# =======================
# GET RECOMMENDATIONS
# =======================
def get_recommendations(disease, severity_level, cough_days=0, language='en', severity_reasons=None):
    """
    Get detailed recommendations based on disease, severity, and specific factors,
    including food recommendations.
    """
    try:
        recommendations = {
            "Normal_Cold": {
                "en": {
                    0: "Mild common cold. Rest at home, drink plenty of fluids, use over-the-counter cold medications if needed. Symptoms should improve in 3-5 days.",
                    1: "Moderate cold symptoms. Get plenty of rest, stay hydrated, use steam inhalation for congestion. Consider consulting a doctor if symptoms persist beyond 7 days.",
                    2: "Severe cold symptoms. May indicate complications like sinusitis or secondary infection. Consult a doctor within 24-48 hours."
                },
                "ta": {
                    0: "லேசான சாதாரண குளிர். வீட்டில் ஓய்வெடுக்கவும், ஏராளமான திரவங்கள் குடிக்கவும், தேவைப்பட்டால் மருந்தக மருந்துகளைப் பயன்படுத்தவும். அறிகுறிகள் 3-5 நாட்களில் மேம்பட வேண்டும்.",
                    1: "மிதமான குளிர் அறிகுறிகள். நன்றாக ஓய்வெடுத்து நீரேற்றமாக இருங்கள், நெரிசலுக்கு நீராவி பிடிக்கவும். 7 நாட்களுக்கு மேல் அறிகுறிகள் நீடித்தால் மருத்துவரை அணுகவும்.",
                    2: "கடுமையான குளிர் அறிகுறிகள். சைனஸ் தொற்று அல்லது இரண்டாம் நிலை தொற்று போன்ற சிக்கல்களைக் குறிக்கலாம். 24-48 மணி நேரத்திற்குள் மருத்துவரை அணுகவும்."
                },
                "food": {
                    "en": "Warm soups, ginger tea with honey, citrus fruits (Vitamin C), garlic, and plenty of warm water.",
                    "ta": "சூடான சூப், தேனுடன் கூடிய இஞ்சி டீ, சிட்ரஸ் பழங்கள் (வைட்டமின் சி), பூண்டு மற்றும் ஏராளமான வெதுவெதுப்பான நீர்."
                }
            },
            "COVID": {
                "en": {
                    0: "Mild COVID-like symptoms. Isolate at home, monitor oxygen levels, rest, stay hydrated. Follow local health guidelines. If symptoms worsen, consult doctor.",
                    1: "Moderate COVID symptoms. Isolate immediately, monitor oxygen saturation, take paracetamol for fever. Consult doctor within 24 hours. Watch for breathing difficulty.",
                    2: "Severe COVID symptoms - SEEK IMMEDIATE MEDICAL ATTENTION. Go to emergency room or call ambulance. Do not wait."
                },
                "ta": {
                    0: "லேசான கோவிட் போன்ற அறிகுறிகள். வீட்டில் தனிமைப்படுத்துங்கள், ஆக்சிஜன் அளவைக் கண்காணிக்கவும், ஓய்வெடுக்கவும், நீரேற்றமாக இருக்கவும். உள்ளூர் சுகாதார வழிகாட்டுதல்களைப் பின்பற்றவும். அறிகுறிகள் மோசமடைந்தால், மருத்துவரை அணுகவும்.",
                    1: "மிதமான கோவிட் அறிகுறிகள். உடனடியாக தனிமைப்படுத்துங்கள், ஆக்சிஜன் செறிவைக் கண்காணிக்கவும், காய்ச்சலுக்கு பாராசிட்டமால் எடுத்துக் கொள்ளுங்கள். 24 மணி நேரத்திற்குள் மருத்துவரை அணுகவும். சுவாசிப்பதில் சிரமம் ஏற்பட்டால் கவனிக்கவும்.",
                    2: "கடுமையான கோவிட் அறிகுறிகள் - உடனடி மருத்துவ உதவியை நாடுங்கள். அவசர சிகிச்சை பிரிவுக்குச் செல்லுங்கள் அல்லது ஆம்புலன்ஸை அழைக்கவும். காத்திருக்க வேண்டாம்."
                },
                "food": {
                    "en": "Protein-rich foods (eggs, dal), zinc-rich foods, Vitamin D, green leafy vegetables, and light, easily digestible meals.",
                    "ta": "புரதம் நிறைந்த உணவுகள் (முட்டை, பருப்பு), துத்தநாகம் நிறைந்த உணவுகள், வைட்டமின் டி, பச்சை இலை காய்கறிகள் மற்றும் எளிதில் ஜீரணிக்கக்கூடிய லேசான உணவுகள்."
                }
            },
            "Asthma": {
                "en": {
                    0: "Mild asthma symptoms. Use rescue inhaler as needed. Avoid triggers (dust, pollen, smoke). Consider daily controller medication if prescribed.",
                    1: "Moderate asthma symptoms. Use rescue inhaler. If not improving within 15-20 minutes, seek medical help. Review asthma action plan.",
                    2: "SEVERE ASTHMA ATTACK - EMERGENCY. Use emergency inhaler/spacer. If no improvement in 5-10 minutes, call ambulance immediately."
                },
                "ta": {
                    0: "லேசான ஆஸ்துமா அறிகுறிகள். தேவைக்கேற்ப மீட்பு உள்ளிழுப்பியைப் பயன்படுத்தவும். தூண்டுதல்களைத் தவிர்க்கவும் (தூசி, மகரந்தம், புகை). மருத்துவர் பரிந்துரைத்தால் தினசரி கட்டுப்பாட்டு மருந்தைப் பயன்படுத்தவும்.",
                    1: "மிதமான ஆஸ்துமா அறிகுறிகள். மீட்பு உள்ளிழுப்பியைப் பயன்படுத்தவும். 15-20 நிமிடங்களில் மேம்படவில்லை என்றால், மருத்துவ உதவியை நாடுங்கள். ஆஸ்துமா செயல் திட்டத்தை மதிப்பாய்வு செய்யவும்.",
                    2: "கடுமையான ஆஸ்துமா தாக்குதல் - அவசரநிலை. அவசர உள்ளிழுப்பி/ஸ்பேசரைப் பயன்படுத்தவும். 5-10 நிமிடங்களில் முன்னேற்றம் இல்லை என்றால், உடனடியாக ஆம்புலன்ஸை அழைக்கவும்."
                },
                "food": {
                    "en": "Foods rich in Omega-3 (flaxseeds, walnuts), apples, bananas, carrots, and magnesium-rich foods like spinach.",
                    "ta": "ஒமேகா-3 நிறைந்த உணவுகள் (ஆளிவிதை, அக்ரூட் பருப்புகள்), ஆப்பிள்கள், வாழைப்பழங்கள், கேரட் மற்றும் கீரை போன்ற மெக்னீசியம் நிறைந்த உணவுகள்."
                }
            },
            "Bronchitis": {
                "en": {
                    0: "Mild bronchitis. Rest, drink warm fluids, use honey for cough, steam inhalation. Avoid smoke. Should improve in 1-2 weeks.",
                    1: "Moderate bronchitis. Consult doctor for possible antibiotics (if bacterial) or cough medications. Use humidifier. Rest adequately.",
                    2: "Severe bronchitis with persistent cough/fever. Seek medical attention promptly. May require prescription medications and monitoring."
                },
                "ta": {
                    0: "லேசான மூச்சுக்குழாய் அழற்சி. ஓய்வு, வெதுவெதுப்பான திரவங்கள் குடிக்கவும், இருமலுக்குத் தேன் பயன்படுத்தவும், நீராவி பிடிக்கவும். புகையைத் தவிர்க்கவும். 1-2 வாரங்களில் மேம்பட வேண்டும்.",
                    1: "மிதமான மூச்சுக்குழாய் அழற்சி. நுண்ணுயிர் எதிர்ப்பிகள் (பாக்டீரியா என்றால்) அல்லது இருமல் மருந்துகளுக்கு மருத்துவரை அணுகவும். ஈரப்பதமாக்கியைப் பயன்படுத்தவும். போதுமான ஓய்வு எடுக்கவும்.",
                    2: "தொடர்ந்து இருமல்/காய்ச்சலுடன் கூடிய கடுமையான மூச்சுக்குழாய் அழற்சி. உடனடியாக மருத்துவ உதவியை நாடுங்கள். மருந்துகள் மற்றும் கண்காணிப்பு தேவைப்படலாம்."
                },
                "food": {
                    "en": "Warm lemonade with honey, almonds, turmeric milk, and antioxidant-rich berries to reduce inflammation.",
                    "ta": "தேனுடன் கூடிய வெதுவெதுப்பான எலுமிச்சை பழச்சாறு, பாதாம், மஞ்சள் பால் மற்றும் வீக்கத்தைக் குறைக்க ஆக்ஸிஜனேற்ற நிறைந்த பெர்ரி பழங்கள்."
                }
            }
        }
        
        lang = language if language in ['en', 'ta'] else 'en'
        disease_key = disease if disease in recommendations else "Normal_Cold"
        severity_key = severity_level if severity_level in [0, 1, 2] else 0
        
        base_recommendation = recommendations[disease_key][lang][severity_key]
        food_recommendation = recommendations[disease_key]["food"][lang]
        
        # Add specific advice based on severity reasons
        if severity_reasons and len(severity_reasons) > 0:
            specific_advice = " " + " ".join(severity_reasons[:2])
            base_recommendation += specific_advice
        
        return {
            "general": base_recommendation,
            "food": food_recommendation
        }
        
    except Exception as e:
        logger.error(f"❌ Recommendations error: {str(e)}")
        if language == 'ta':
            return {"general": "மருத்துவ ஆலோசனைக்கு சுகாதார நிபுணரை அணுகவும்.", "food": ""}
        return {"general": "Please consult a healthcare professional for medical advice.", "food": ""}

# =======================
# REAL-WORLD HOSPITAL DATA (TAMIL NADU)
# =======================
REAL_WORLD_HOSPITALS = [
    {
        "name": "Apollo Hospitals, Greams Road, Chennai",
        "lat": 13.0631, "lng": 80.2526,
        "specialists": {
            "COVID": ["Dr. Narasimhan R (Pulmonologist)", "Dr. Sundararajan L"],
            "Asthma": ["Dr. Raj B Singh (Respirologist)", "Dr. Ilangho R P"],
            "Bronchitis": ["Dr. A R Gayathri Devi", "Dr. Babu K Abraham"],
            "Normal_Cold": ["General Physicians"]
        },
        "timings": "24/7",
        "phone": "1860-500-1066",
        "map_url": "https://www.google.com/maps/search/Apollo+Hospitals+Greams+Road+Chennai",
        "expertise": ["COVID", "Asthma", "Bronchitis", "Normal_Cold"]
    },
    {
        "name": "MIOT International, Chennai",
        "lat": 13.0189, "lng": 80.1873,
        "specialists": {
            "COVID": ["MIOT COVID Care Specialists"],
            "Bronchitis": ["Infectious Disease Specialists"],
            "Normal_Cold": ["General Medicine Doctors"]
        },
        "timings": "24/7",
        "phone": "+91 44 42002288",
        "map_url": "https://www.google.com/maps/search/MIOT+International+Chennai",
        "expertise": ["COVID", "Bronchitis", "Normal_Cold"]
    },
    {
        "name": "Kovai Medical Center (KMCH), Coimbatore",
        "lat": 11.0428, "lng": 77.0360,
        "specialists": {
            "Asthma": ["KMCH Respiratory Care Unit"],
            "Bronchitis": ["Pulmonology Dept Specialists"],
            "COVID": ["Post-COVID Special Care Team"],
            "Normal_Cold": ["Internal Medicine Dept"]
        },
        "timings": "24/7",
        "phone": "+91-422-4323800",
        "map_url": "https://www.google.com/maps/search/Kovai+Medical+Center+Coimbatore",
        "expertise": ["Asthma", "Bronchitis", "COVID", "Normal_Cold"]
    },
    {
        "name": "Meenakshi Mission Hospital, Madurai",
        "lat": 9.9485, "lng": 78.1625,
        "specialists": {
            "COVID": ["Critical Care Pulmonologists"],
            "Asthma": ["Dr. Vel Kumar Gopal (Pulmonologist)"],
            "Bronchitis": ["Respiratory Disease Specialists"],
            "Normal_Cold": ["General Medicine Unit"]
        },
        "timings": "24/7",
        "phone": "+91-452-4263000",
        "map_url": "https://www.google.com/maps/search/Meenakshi+Mission+Hospital+Madurai",
        "expertise": ["COVID", "Asthma", "Bronchitis", "Normal_Cold"]
    },
    {
        "name": "Kauvery Hospital, Trichy",
        "lat": 10.8193, "lng": 78.6865,
        "specialists": {
            "COVID": ["Kauvery COVID Specialists"],
            "Bronchitis": ["Pulmonology Team"],
            "Asthma": ["Respiratory Care Specialists"],
            "Normal_Cold": ["General Outpatient Dept"]
        },
        "timings": "24/7",
        "phone": "0431-4077777",
        "map_url": "https://www.google.com/maps/search/Kauvery+Hospital+Tennur+Trichy",
        "expertise": ["COVID", "Bronchitis", "Asthma", "Normal_Cold"]
    },
    {
        "name": "Shifa Hospital, Tirunelveli Junction",
        "lat": 8.7306, "lng": 77.7128,
        "specialists": {
            "Asthma": ["Dr. Bala (Pulmonologist)", "Asthma Care Team"],
            "Bronchitis": ["Dr. Prince Sudharsan (Chest Physician)"],
            "COVID": ["Pulmonary Medicine Unit"],
            "Normal_Cold": ["General Outpatient Services"]
        },
        "timings": "24/7",
        "phone": "0462-2323041",
        "map_url": "https://www.google.com/maps/search/Shifa+Hospital+Tirunelveli",
        "expertise": ["Asthma", "Bronchitis", "COVID", "Normal_Cold"]
    },
    {
        "name": "Annai Velankanni Hospital, Tirunelveli",
        "lat": 8.7171, "lng": 77.7380,
        "specialists": {
            "Asthma": ["Dr. O.M. Mohideen Haji (Pulmonologist)"],
            "Bronchitis": ["Respiratory Care Specialists"],
            "COVID": ["Emergency Pulmonary Care"],
            "Normal_Cold": ["Infectious Disease Dept"]
        },
        "timings": "24/7",
        "phone": "+91-9077919191",
        "map_url": "https://www.google.com/maps/search/Annai+Velankanni+Hospital+Tirunelveli",
        "expertise": ["Asthma", "Bronchitis", "COVID", "Normal_Cold"]
    },
    {
        "name": "Galaxy Hospital, Vannarpettai, Tirunelveli",
        "lat": 8.7259, "lng": 77.7286,
        "specialists": {
            "COVID": ["Specialized Pulmonology Unit"],
            "Asthma": ["Respiratory Medicine Team"],
            "Bronchitis": ["Chest Disease Specialists"],
            "Normal_Cold": ["General Medicine Dept"]
        },
        "timings": "24/7",
        "phone": "0462-2501951",
        "map_url": "https://www.google.com/maps/search/Galaxy+Hospital+Vannarpettai+Tirunelveli",
        "expertise": ["COVID", "Asthma", "Bronchitis", "Normal_Cold"]
    },
    {
        "name": "Tirunelveli Medical College Hospital (TVMCH)",
        "lat": 8.7145, "lng": 77.7553,
        "specialists": {
            "COVID": ["Dept of Thoracic Medicine"],
            "Bronchitis": ["Pulmonology Ward Specialists"],
            "Asthma": ["Respiratory Care Govt Unit"],
            "Normal_Cold": ["Community Health Outpatient"]
        },
        "timings": "24/7",
        "phone": "0462-2572944",
        "map_url": "https://www.google.com/maps/search/Tirunelveli+Medical+College+Hospital",
        "expertise": ["COVID", "Bronchitis", "Asthma", "Normal_Cold"]
    },
    {
        "name": "Salem Gopi Hospital, Salem",
        "lat": 11.6746, "lng": 78.1515,
        "specialists": {
            "Asthma": ["Pulmonary Function Lab Specialists"],
            "Bronchitis": ["Respiratory Medicine Team"],
            "COVID": ["Gopi Hospital COVID Care"],
            "Normal_Cold": ["General Medicine Dept"]
        },
        "timings": "24/7",
        "phone": "0427-2666444",
        "map_url": "https://www.google.com/maps/search/Salem+Gopi+Hospital",
        "expertise": ["Asthma", "Bronchitis", "COVID", "Normal_Cold"]
    },
    {
        "name": "Christian Medical College (CMC), Vellore",
        "lat": 12.9249, "lng": 79.1347,
        "specialists": {
            "COVID": ["CMC Infectious Disease Dept"],
            "Asthma": ["CMC Pulmonology Specialists"],
            "Bronchitis": ["Respiratory Medicine Team"],
            "Normal_Cold": ["Community Health Dept"]
        },
        "timings": "24/7",
        "phone": "0416-2281000",
        "map_url": "https://www.google.com/maps/search/Christian+Medical+College+Vellore",
        "expertise": ["COVID", "Asthma", "Bronchitis", "Normal_Cold"]
    }
]

def calculate_distance(lat1, lon1, lat2, lon2):
    """Haversine formula to calculate distance in km"""
    try:
        R = 6371  # Earth radius in km
        dLat = math.radians(float(lat2) - float(lat1))
        dLon = math.radians(float(lon2) - float(lon1))
        a = math.sin(dLat/2) * math.sin(dLat/2) + \
            math.cos(math.radians(float(lat1))) * math.cos(math.radians(float(lat2))) * \
            math.sin(dLon/2) * math.sin(dLon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    except:
        return 9999

def fetch_dynamic_hospitals(lat, lng, radius_km=30):
    """Fetch hospitals from OpenStreetMap using Overpass API"""
    logger.info(f"🌐 Fetching dynamic hospitals near {lat}, {lng}...")
    
    # Overpass QL query: find nodes/ways/relations tagged as hospital within 'radius' meters
    # We use a 30km radius by default
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="hospital"](around:{radius_km * 1000},{lat},{lng});
      way["amenity"="hospital"](around:{radius_km * 1000},{lat},{lng});
      relation["amenity"="hospital"](around:{radius_km * 1000},{lat},{lng});
    );
    out center;
    """
    
    try:
        response = requests.post(overpass_url, data={'data': overpass_query}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            dynamic_hospitals = []
            for element in data.get('elements', []):
                name = element.get('tags', {}).get('name')
                if not name:
                    name = "Medical Facility"
                
                # Get coordinates (center for ways/relations, lat/lon for nodes)
                h_lat = element.get('lat') or element.get('center', {}).get('lat')
                h_lng = element.get('lon') or element.get('center', {}).get('lon')
                
                if h_lat and h_lng:
                    phone = element.get('tags', {}).get('phone') or element.get('tags', {}).get('contact:phone') or "Contact via Maps"
                    website = element.get('tags', {}).get('website') or ""
                    
                    dynamic_hospitals.append({
                        "name": name,
                        "lat": h_lat,
                        "lng": h_lng,
                        "phone": phone,
                        "website": website,
                        "map_url": f"https://www.google.com/maps/search/?api=1&query={h_lat},{h_lng}",
                        "is_verified": False,
                        "specialists": {}, # Will be filled if it matches a verified one
                        "timings": element.get('tags', {}).get('opening_hours', "Check Website/Call"),
                        "expertise": ["General Medical Care"]
                    })
            logger.info(f"✅ Found {len(dynamic_hospitals)} hospitals via Overpass API")
            return dynamic_hospitals
    except Exception as e:
        logger.error(f"❌ Error fetching dynamic hospitals: {str(e)}")
    
    return []

def get_nearby_hospitals(user_lat, user_lng, disease):
    """Find hospitals specializing in disease near user (Hybrid: Static + Dynamic)"""
    if user_lat is None or user_lng is None:
        return []
    
    try:
        user_lat = float(user_lat)
        user_lng = float(user_lng)
    except:
        return []

    combined_results = []
    seen_names = set()

    # 1. Check Verified Static Database first (High Quality)
    for h in REAL_WORLD_HOSPITALS:
        dist = calculate_distance(user_lat, user_lng, h['lat'], h['lng'])
        # Store verified status
        h_copy = h.copy()
        h_copy['distance'] = round(dist, 2)
        h_copy['is_verified'] = True
        
        # Pull disease-specific specialists if any
        h_copy['specialists'] = h['specialists'].get(disease, ["Specialist Team"])
        
        combined_results.append(h_copy)
        seen_names.add(h['name'].lower())

    # 2. Fetch Dynamic results from OpenStreetMap (Global Coverage)
    dynamic_hospitals = fetch_dynamic_hospitals(user_lat, user_lng)
    
    for dh in dynamic_hospitals:
        # Avoid duplicates if the dynamic search finds one of our hardcoded ones
        is_duplicate = False
        for name in seen_names:
            if name in dh['name'].lower() or dh['name'].lower() in name:
                is_duplicate = True
                break
        
        if not is_duplicate:
            dist = calculate_distance(user_lat, user_lng, dh['lat'], dh['lng'])
            dh['distance'] = round(dist, 2)
            combined_results.append(dh)

    # Sort by distance
    combined_results.sort(key=lambda x: x['distance'])
    
    # Return top 5
    return combined_results[:5]

# =======================
# SIMPLE PREDICTION FALLBACK
# =======================
def simple_prediction_fallback(symptoms_dict, age, cough_days):
    """
    Simple rule-based prediction as fallback
    """
    try:
        # Count symptoms
        fever = 1 if symptoms_dict.get('fever', 'no') == 'yes' else 0
        dry_cough = 1 if symptoms_dict.get('dry_cough', 'no') == 'yes' else 0
        wet_cough = 1 if symptoms_dict.get('wet_cough', 'no') == 'yes' else 0
        wheezing = 1 if symptoms_dict.get('wheezing', 'no') == 'yes' else 0
        breath = 1 if symptoms_dict.get('breath', 'no') == 'yes' else 0
        chest = 1 if symptoms_dict.get('chest', 'no') == 'yes' else 0
        smell = 1 if symptoms_dict.get('smell', 'no') == 'yes' else 0
        smoker = 1 if symptoms_dict.get('smoker', 'no') == 'yes' else 0
        fatigue = 1 if symptoms_dict.get('fatigue', 'no') == 'yes' else 0
        sore_throat = 1 if symptoms_dict.get('sore_throat', 'no') == 'yes' else 0
        
        # Refined Logic
        # 1. COVID: Strong indicators only
        if smell == 1 or (fever == 1 and dry_cough == 1 and breath == 1):
             return "COVID", 0.85
             
        # 2. Asthma: Wheezing is key
        elif wheezing == 1 and (breath == 1 or chest == 1) and fever == 0:
            return "Asthma", 0.80
            
        # 3. Bronchitis: Wet cough + long duration + smoker/chronic
        elif (wet_cough == 1 and cough_days > 14) or (wet_cough == 1 and smoker == 1 and cough_days > 7):
            return "Bronchitis", 0.75
            
        # 4. Default to Normal Cold for common symptoms
        else:
            return "Normal_Cold", 0.85
            
    except Exception as e:
        logger.error(f"❌ Simple prediction fallback error: {str(e)}")
        return "Normal_Cold", 0.60

# =======================
# FIREBASE HELPER FUNCTIONS
# =======================
def verify_user_token(token):
    """Verify Firebase ID token"""
    try:
        if not token:
            return None
        
        decoded_token = auth.verify_id_token(token)
        return decoded_token['uid']
    except Exception as e:
        logger.error(f"Token verification failed: {str(e)}")
        return None

def save_screening_to_firestore(user_id, patient_name, result_data, symptoms_data, age, cough_days):
    """Save screening result to Firestore"""
    try:
        if db is None:
            logger.warning("Firestore not available, skipping save")
            return
        
        screening_data = {
            'userId': user_id,
            'patientName': patient_name,
            'age': age,
            'coughDays': cough_days,
            'symptoms': symptoms_data,
            'disease': result_data.get('disease'),
            'severity': result_data.get('severity_en'),
            'severity_code': result_data.get('severity_code'),
            'recommendation': result_data.get('recommendation_en'),
            'confidence': result_data.get('confidence'),
            'covidAnalysis': result_data.get('covid_analysis_en'),
            'timestamp': firestore.SERVER_TIMESTAMP
        }
        
        # Add to screenings collection
        db.collection('screenings').add(screening_data)
        
        # Update user's family members
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        
        if user_doc.exists:
            current_members = user_doc.to_dict().get('familyMembers', [])
            if patient_name not in current_members:
                user_ref.update({
                    'familyMembers': firestore.ArrayUnion([patient_name])
                })
        
        logger.info(f"✅ Screening saved to Firestore for user: {user_id}")
        
    except Exception as e:
        logger.error(f"❌ Error saving to Firestore: {str(e)}")

# =======================
# MAIN PREDICTION ENDPOINT
# =======================
@app.route("/predict", methods=["POST", "GET", "OPTIONS"])
def predict():
    """
    Main prediction endpoint with improved logic:
    - Handles CORS preflight (OPTIONS)
    - Uses ML models if available and confident
    - Falls back to rule-based if models fail or low confidence
    - Audio analysis uses mel spectrograms (matching training)
    - Comprehensive severity assessment
    """
    # Handle preflight request explicitly
    if request.method == "OPTIONS":
        origin = request.headers.get("Origin")
        response = make_response()
        if origin in ALLOWED_ORIGINS:
            response.headers.add("Access-Control-Allow-Origin", origin)
        else:
            response.headers.add("Access-Control-Allow-Origin", "*")
            
        response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization,Accept,Origin,X-Requested-With")
        response.headers.add('Access-Control-Allow-Methods', "GET,PUT,POST,DELETE,OPTIONS")
        response.headers.add('Access-Control-Allow-Credentials', "true")
        return response
        
    try:
        logger.info(f"📥 Received {request.method} prediction request from {request.remote_addr}")
        logger.info(f"Headers: {dict(request.headers)}")
        
        # Handle GET requests for testing
        if request.method == "GET":
            return jsonify({
                "success": True,
                "message": "Prediction endpoint is active. Send POST request with symptoms data.",
                "required_fields": ["symptoms", "age", "cough_days", "language"],
                "example_symptoms": {
                    "fever": "yes",
                    "dry_cough": "yes",
                    "wet_cough": "no",
                    "wheezing": "no",
                    "breath": "no",
                    "chest": "no",
                    "fatigue": "yes",
                    "sore_throat": "yes",
                    "smell": "yes",
                    "night_cough": "no",
                    "smoker": "no",
                    "chronic": "no"
                }
            })
        
        # Handle POST requests
        symptoms = {}
        age = 0
        cough_days = 0
        language = "en"
        user_id = None
        patient_name = None
        
        # Check content type
        content_type = request.content_type or ""
        
        if 'application/json' in content_type:
            # JSON data
            data = request.get_json()
            symptoms = data.get("symptoms", {})
            age = int(data.get("age", 0))
            cough_days = int(data.get("cough_days", 0))
            language = data.get("language", "en")
            user_id = data.get("userId")
            patient_name = data.get("patientName")
            lat = data.get("lat")
            lng = data.get("lng")
            
        elif 'multipart/form-data' in content_type or 'application/x-www-form-urlencoded' in content_type:
            # Form data
            symptoms_str = request.form.get("symptoms", "{}")
            try:
                symptoms = json.loads(symptoms_str)
            except:
                symptoms = {}
            
            try:
                age = int(request.form.get("age", 0))
            except:
                age = 0
            
            try:
                cough_days = int(request.form.get("cough_days", 0))
            except:
                cough_days = 0
            
            language = request.form.get("language", "en")
            user_id = request.form.get("userId")
            patient_name = request.form.get("patientName")
            lat = request.form.get("lat")
            lng = request.form.get("lng")
        
        else:
            # Try to parse as JSON anyway
            try:
                data = request.get_json()
                if data:
                    symptoms = data.get("symptoms", {})
                    age = int(data.get("age", 0))
                    cough_days = int(data.get("cough_days", 0))
                    language = data.get("language", "en")
                    user_id = data.get("userId")
                    patient_name = data.get("patientName")
                    lat = data.get("lat")
                    lng = data.get("lng")
            except Exception as e:
                logger.debug(f"Location parsing skipped: {str(e)}")
        
        if lat and lng:
            logger.info(f"📍 Received Location: {lat}, {lng}")
        else:
            logger.info("📍 No location received from client")
        
        log_memory("Start Prediction Request")
        
        # ===== AUDIO PROCESSING =====
        audio_features = None
        audio_prediction = None
        audio_confidence = 0.0
        audio_covid_score = 0
        
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file and audio_file.filename != '':
                logger.info(f"🎤 Audio file received: {audio_file.filename}")
                
                # Check for Lite Mode
                disable_heavy_ml = os.environ.get('RENDER_LITE_MODE', 'false').lower() == 'true'
                
                # ONE-TIME LOAD: This is the memory-critical part
                try:
                    logger.info("📡 Loading audio into memory (one-time)...")
                    y, sr = safe_process_audio(audio_file)
                    
                    if y is not None:
                        # Extract features for rules using the already loaded data
                        if not disable_heavy_ml:
                            audio_features = extract_audio_features_for_rules(y, sr)
                        
                        # Process for COVID model using the already loaded data
                        if covid_model and not disable_heavy_ml:
                            try:
                                logger.info("🧠 Running optimized audio ML...")
                                mel_spectrogram = process_audio_for_covid(y, sr)
                                
                                if mel_spectrogram is not None:
                                    # Make prediction
                                    audio_prediction, audio_confidence, audio_covid_score = predict_from_audio(mel_spectrogram)
                                    
                                    # AGGRESSIVE CLEANUP
                                    del mel_spectrogram
                                    gc.collect()
                                    log_memory("After ML Prediction Cleanup")
                            except Exception as ml_err:
                                logger.error(f"⚠️ Heavy ML failed: {str(ml_err)}")
                        
                        # CLEAN UP THE MAIN AUDIO ARRAY IMMEDIATELY
                        del y
                        gc.collect()
                        log_memory("After Audio Data Cleanup")
                    else:
                        logger.warning("⚠️ Could not load audio data from file")
                        
                except Exception as load_err:
                    logger.error(f"❌ Audio pipeline crashed during load: {str(load_err)}")
        
        # ===== SYMPTOM-BASED PREDICTION =====
        ml_prediction = None
        ml_confidence = 0.0
        ml_probabilities = None
        rule_prediction = None
        rule_confidence = 0.0
        rule_scores = None
        
        # Try ML model first
        if disease_model and NUMPY_AVAILABLE:
            try:
                features = enhance_features(symptoms, age, cough_days)
                if features is not None:
                    # Scale features
                    if disease_scaler:
                        features_scaled = disease_scaler.transform(features)
                    else:
                        features_scaled = features
                    
                    # Make prediction
                    raw_prediction = disease_model.predict(features_scaled)[0]
                    
                    # Get probabilities if available
                    if hasattr(disease_model, 'predict_proba'):
                        proba = disease_model.predict_proba(features_scaled)[0]
                        ml_confidence = float(max(proba))
                        ml_probabilities = proba.tolist()
                        
                        # Map prediction to class
                        if hasattr(disease_model, 'classes_'):
                            classes = disease_model.classes_
                            pred_idx = np.argmax(proba)
                            ml_prediction = classes[pred_idx]
                        else:
                            ml_prediction = raw_prediction
                        
                        logger.info(f"📊 ML Probabilities: {dict(zip(classes, proba))}")
                    else:
                        ml_prediction = raw_prediction
                        ml_confidence = 0.75
                    
                    logger.info(f"🎯 ML Prediction: {ml_prediction} (confidence: {ml_confidence:.2f})")
            except Exception as e:
                logger.error(f"❌ ML prediction failed: {str(e)}")
        
        # Always run rule-based for comparison
        rule_prediction, rule_confidence, rule_scores, rule_reasoning = rule_based_disease_detection(
            symptoms, age, cough_days, audio_features
        )
        
        # ===== DECISION LOGIC =====
        # Decide which prediction to use
        final_disease = rule_prediction
        final_confidence = rule_confidence
        prediction_source = "rule-based"
        
        # Use ML if confident enough
        if ml_prediction and ml_confidence > 0.7:
            final_disease = ml_prediction
            final_confidence = ml_confidence
            prediction_source = "ml"
            logger.info(f"✅ Using ML prediction (confidence > 0.7)")
        
        # If ML confidence is moderate, blend with rule-based
        elif ml_prediction and ml_confidence > 0.5:
            # Check if they agree
            if ml_prediction == rule_prediction:
                final_disease = ml_prediction
                final_confidence = (ml_confidence + rule_confidence) / 2
                prediction_source = "blended-agree"
                logger.info(f"✅ ML and rule-based agree, using blended confidence")
            else:
                # They disagree - use the one with higher confidence
                if ml_confidence > rule_confidence:
                    final_disease = ml_prediction
                    final_confidence = ml_confidence * 0.9  # Slightly reduce confidence due to disagreement
                    prediction_source = "ml-higher"
                else:
                    final_disease = rule_prediction
                    final_confidence = rule_confidence
                    prediction_source = "rule-higher"
                logger.info(f"⚠️ ML and rule-based disagree, using {prediction_source}")
        
        # ===== AUDIO-BASED COVID ENHANCEMENT =====
        # If audio strongly indicates COVID, override for COVID cases
        if audio_prediction == "COVID" and audio_confidence > 0.7:
            logger.info(f"🎤 Strong audio COVID signal (conf: {audio_confidence:.2f})")
            
            if final_disease != "COVID":
                # Audio suggests COVID but symptoms don't - investigate
                if symptoms.get('smell', 'no') == 'yes' or symptoms.get('fever', 'no') == 'yes' or symptoms.get('breath', 'no') == 'yes':
                    # Symptoms support COVID possibility
                    final_disease = "COVID"
                    final_confidence = (final_confidence + audio_confidence) / 2
                    prediction_source = "audio-enhanced"
                    logger.info(f"✅ Audio overrode to COVID based on supporting symptoms")
        
        # ===== SEVERITY ASSESSMENT =====
        severity_level, severity_text, severity_score, severity_reasons = assess_severity(
            final_disease, symptoms, age, cough_days, audio_features
        )
        
        # ===== COVID ANALYSIS TEXT =====
        covid_analysis = "Not analyzed"
        if final_disease == "COVID":
            covid_analysis = "Symptoms and "
            if audio_prediction == "COVID":
                covid_analysis += "mel spectrogram analysis indicates COVID patterns"
            else:
                covid_analysis += "symptom pattern suggests COVID"
        elif audio_prediction == "COVID":
            covid_analysis = "Mel spectrogram shows COVID-like patterns but symptoms suggest otherwise"
        else:
            covid_analysis = "No strong COVID indicators"
        
        # ===== GET RECOMMENDATIONS =====
        rec_data_en = get_recommendations(
            final_disease, severity_level, cough_days, 'en', severity_reasons
        )
        rec_data_ta = get_recommendations(
            final_disease, severity_level, cough_days, 'ta', severity_reasons
        )
        
        recommendation_en = rec_data_en['general']
        food_recommendation_en = rec_data_en['food']
        
        recommendation_ta = rec_data_ta['general']
        food_recommendation_ta = rec_data_ta['food']
        
        # ===== GET NEARBY HOSPITALS =====
        nearby_hospitals = get_nearby_hospitals(lat, lng, final_disease)
        
        # ===== PREPARE RESPONSE =====
        response = {
            "success": True,
            "disease": final_disease,
            "severity_en": severity_text,
            "severity_ta": "கடுமையான" if severity_text == "Severe" else ("மிதமான" if severity_text == "Moderate" else "லேசான"),
            "severity_code": severity_level,
            "recommendation_en": recommendation_en,
            "recommendation_ta": recommendation_ta,
            "food_recommendation_en": food_recommendation_en,
            "food_recommendation_ta": food_recommendation_ta,
            "nearby_hospitals": nearby_hospitals,
            "confidence": f"{final_confidence*100:.1f}%",
            "covid_analysis_en": covid_analysis,
            "covid_analysis_ta": "மெல் ஸ்பெக்ட்ரோகிராம் பகுப்பாய்வின் அடிப்படையில்",
            "analysis_details": {
                "source": prediction_source,
                "ml_prediction": ml_prediction,
                "ml_confidence": f"{ml_confidence*100:.1f}%" if ml_confidence else None,
                "rule_prediction": rule_prediction,
                "rule_confidence": f"{rule_confidence*100:.1f}%" if rule_confidence else None,
                "audio_prediction": audio_prediction,
                "audio_confidence": f"{audio_confidence*100:.1f}%" if audio_confidence else None,
                "severity_score": severity_score,
                "severity_factors": severity_reasons
            }
        }
        
        # Save to Firestore if user is logged in
        if user_id and patient_name:
            save_screening_to_firestore(user_id, patient_name, response, symptoms, age, cough_days)
        
        logger.info(f"✅ Final: {final_disease} | Severity: {severity_text} | Confidence: {final_confidence:.1%} | Source: {prediction_source}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Prediction endpoint error: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Prediction failed. Please check server logs."
        }), 500

# =======================
# USER MANAGEMENT ENDPOINTS
# =======================
@app.route("/api/user/profile", methods=["GET"])
def get_user_profile():
    """Get user profile"""
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        user_id = verify_user_token(token)
        
        if not user_id:
            return jsonify({"success": False, "error": "Invalid token"}), 401
        
        if db is None:
            return jsonify({"success": False, "error": "Database not available"}), 500
        
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            return jsonify({"success": False, "error": "User not found"}), 404
        
        user_data = user_doc.to_dict()
        
        # Get screening count
        screenings_ref = db.collection('screenings').where('userId', '==', user_id)
        screening_count = len(list(screenings_ref.stream()))
        
        response = {
            "success": True,
            "user": {
                "id": user_id,
                "email": user_data.get('email'),
                "name": user_data.get('name'),
                "familyMembers": user_data.get('familyMembers', []),
                "createdAt": user_data.get('createdAt'),
                "lastLogin": user_data.get('lastLogin'),
                "screeningCount": screening_count
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting user profile: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/user/screenings", methods=["GET"])
def get_user_screenings():
    """Get user's screening history"""
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        user_id = verify_user_token(token)
        
        if not user_id:
            return jsonify({"success": False, "error": "Invalid token"}), 401
        
        if db is None:
            return jsonify({"success": False, "error": "Database not available"}), 500
        
        limit = request.args.get('limit', default=20, type=int)
        patient_name = request.args.get('patient')
        
        screenings_ref = db.collection('screenings').where('userId', '==', user_id)
        
        if patient_name:
            screenings_ref = screenings_ref.where('patientName', '==', patient_name)
        
        screenings_ref = screenings_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
        
        screenings = []
        for doc in screenings_ref.stream():
            data = doc.to_dict()
            data['id'] = doc.id
            # Convert timestamp to string for JSON serialization
            if 'timestamp' in data and hasattr(data['timestamp'], 'isoformat'):
                data['timestamp'] = data['timestamp'].isoformat()
            screenings.append(data)
        
        return jsonify({
            "success": True,
            "screenings": screenings,
            "count": len(screenings)
        })
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error getting screenings: {error_msg}")
        
        if "index" in error_msg.lower() and "requires" in error_msg.lower():
            # This is a common Firestore index error
            return jsonify({
                "success": False, 
                "error": "Database index required. Please contact admin to create the required Firestore index.",
                "details": error_msg
            }), 500
            
        return jsonify({"success": False, "error": error_msg}), 500

# =======================
# SIMPLE ENDPOINTS
# =======================
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "service": "VoiceGuard AI",
        "timestamp": datetime.now().isoformat(),
        "dependencies": {
            "numpy": NUMPY_AVAILABLE,
            "librosa": LIBROSA_AVAILABLE,
            "opencv": CV2_AVAILABLE,
            "tensorflow": TENSORFLOW_AVAILABLE,
            "joblib": JOBLIB_AVAILABLE,
            "scipy": SCIPY_AVAILABLE
        },
        "models": {
            "covid_audio": covid_model is not None,
            "disease_classification": disease_model is not None,
            "disease_scaler": disease_scaler is not None
        },
        "firebase": db is not None,
        "directories": {
            "backend": BASE_DIR,
            "frontend": FRONTEND_DIR,
            "project_root": PROJECT_ROOT
        }
    })

@app.route("/", methods=["GET"])
def serve_frontend():
    """Serve the frontend HTML file"""
    try:
        # Check if index.html exists in frontend directory
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        
        if os.path.exists(index_path):
            return send_from_directory(FRONTEND_DIR, "index.html")
        else:
            # If frontend not found, show setup instructions
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>VoiceGuard AI - Setup Required</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }}
                    .container {{ max-width: 800px; margin: 0 auto; padding: 30px; background: rgba(255,255,255,0.1); border-radius: 20px; }}
                    h1 {{ text-align: center; margin-bottom: 30px; }}
                    .step {{ background: rgba(255,255,255,0.15); padding: 20px; border-radius: 10px; margin-bottom: 15px; }}
                    code {{ background: rgba(0,0,0,0.3); padding: 5px 10px; border-radius: 5px; }}
                    .success {{ color: #4CAF50; font-weight: bold; }}
                    .error {{ color: #ff6b6b; font-weight: bold; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🎤 VoiceGuard AI Backend</h1>
                    
                    <div class="step">
                        <h2>✅ Backend is Running!</h2>
                        <p>Your Flask backend server is successfully running</p>
                    </div>
                    
                    <div class="step">
                        <h2>⚠️ Frontend Setup Required</h2>
                        <p>Frontend file not found at: <code>{index_path}</code></p>
                        
                        <h3>To Fix This:</h3>
                        <ol>
                            <li>Create a folder named <code>frontend</code> in your project root</li>
                            <li>Place your <code>index.html</code> file inside it</li>
                            <li>Your directory structure should look like:</li>
                        </ol>
                        <pre style="background: rgba(0,0,0,0.3); padding: 15px; border-radius: 8px;">
VOICEGUARD_PROJECT/
├── backend/
│   ├── app.py
│   └── ... (backend files)
└── frontend/
    └── index.html  ← Place your HTML file here
                        </pre>
                    </div>
                    
                    <div class="step">
                        <h2>📡 Test Backend APIs</h2>
                        <p>While setting up frontend, you can test backend APIs:</p>
                        <ul>
                            <li><a href="/health" style="color: #4CAF50;">/health</a> - Health check endpoint</li>
                            <li><a href="/test" style="color: #4CAF50;">/test</a> - Test endpoint</li>
                            <li><a href="/predict" style="color: #4CAF50;">/predict</a> - Prediction endpoint (GET for info)</li>
                        </ul>
                    </div>
                    
                    <div class="step">
                        <h2>🔧 Quick Test Command</h2>
                        <p>Test the prediction API with curl:</p>
                        <code>
                        curl -X POST http://localhost:5000/predict \\
                          -H "Content-Type: application/json" \\
                          -d '{{"symptoms": {{"fever": "yes"}}, "age": 30, "cough_days": 5}}'
                        </code>
                    </div>
                </div>
            </body>
            </html>
            """
    except Exception as e:
        logger.error(f"Error serving frontend: {str(e)}")
        return jsonify({"error": "Frontend not available"}), 404

@app.route("/health", methods=["GET"])
def health():
    """Basic health check endpoint"""
    return jsonify({
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "firebase": db is not None,
        "models": {
            "disease": disease_model is not None,
            "covid": covid_model is not None
        }
    }), 200

@app.route("/test", methods=["GET"])
def test():
    """Simple test endpoint"""
    return jsonify({
        "message": "VoiceGuard AI Backend is running!",
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "GET /": "Serve frontend",
            "GET /health": "Health check",
            "POST /predict": "Analyze symptoms",
            "GET /test": "This test endpoint",
            "GET /api/user/profile": "Get user profile (requires auth)",
            "GET /api/user/screenings": "Get user screenings (requires auth)"
        },
        "frontend_status": "Found" if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")) else "Not found"
    })

@app.route("/api/debug/models", methods=["GET"])
def debug_models():
    """Debug endpoint to check model status"""
    status = {
        "covid_model": {
            "loaded": covid_model is not None,
            "type": str(type(covid_model)) if covid_model else None,
            "input_shape": str(covid_model.input_shape) if covid_model and hasattr(covid_model, 'input_shape') else None,
        },
        "disease_model": {
            "loaded": disease_model is not None,
            "type": str(type(disease_model)) if disease_model else None,
            "has_predict_proba": hasattr(disease_model, 'predict_proba') if disease_model else False,
            "classes": list(disease_model.classes_) if disease_model and hasattr(disease_model, 'classes_') else None
        },
        "scaler": disease_scaler is not None,
        "dependencies": {
            "numpy": NUMPY_AVAILABLE,
            "librosa": LIBROSA_AVAILABLE,
            "tensorflow": TENSORFLOW_AVAILABLE,
            "joblib": JOBLIB_AVAILABLE,
            "opencv": CV2_AVAILABLE
        }
    }
    
    # Test with sample data
    if disease_model and NUMPY_AVAILABLE and disease_scaler:
        try:
            test_symptoms = {
                "fever": "yes", "dry_cough": "yes", "wet_cough": "no",
                "wheezing": "no", "breath": "yes", "chest": "no",
                "fatigue": "yes", "sore_throat": "no", "smell": "yes",
                "night_cough": "no", "smoker": "no", "chronic": "no"
            }
            features = enhance_features(test_symptoms, 45, 7)
            if features is not None:
                features_scaled = disease_scaler.transform(features)
                pred = disease_model.predict(features_scaled)[0]
                if hasattr(disease_model, 'predict_proba'):
                    proba = disease_model.predict_proba(features_scaled)[0]
                    status["test_prediction"] = {
                        "disease": str(pred),
                        "probabilities": [float(p) for p in proba],
                        "max_prob": float(max(proba))
                    }
        except Exception as e:
            status["test_error"] = str(e)
    
    return jsonify(status)

# =======================
# RUN SERVER
# =======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    
    print("\n" + "=" * 70)
    print("🎤 VOICEGUARD AI - BACKEND SERVER")
    print("=" * 70)
    print(f"📁 Server Directory: {BASE_DIR}")
    print(f"📁 Project Root: {PROJECT_ROOT}")
    print(f"📁 Frontend Directory: {FRONTEND_DIR}")
    print(f"🌐 Server URL: http://localhost:{port}")
    print("=" * 70)
    
    # Check frontend
    if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
        print("✅ Frontend: FOUND (index.html)")
    else:
        print(f"⚠️  Frontend: NOT FOUND at {FRONTEND_DIR}/index.html")
        print("   Please place your index.html file in the 'frontend' folder")
    
    print("=" * 70)
    print("📦 DEPENDENCY STATUS:")
    print(f"  • NumPy: {'✅ AVAILABLE' if NUMPY_AVAILABLE else '❌ NOT AVAILABLE'}")
    print(f"  • Librosa: {'✅ AVAILABLE' if LIBROSA_AVAILABLE else '❌ NOT AVAILABLE'}")
    print(f"  • SciPy: {'✅ AVAILABLE' if SCIPY_AVAILABLE else '❌ NOT AVAILABLE'}")
    print(f"  • TensorFlow: {'✅ AVAILABLE' if TENSORFLOW_AVAILABLE else '❌ NOT AVAILABLE'}")
    print(f"  • Joblib: {'✅ AVAILABLE' if JOBLIB_AVAILABLE else '❌ NOT AVAILABLE'}")
    print(f"  • OpenCV: {'✅ AVAILABLE' if CV2_AVAILABLE else '❌ NOT AVAILABLE'}")
    print(f"  • Firebase Admin: {'✅ AVAILABLE' if db is not None else '❌ NOT AVAILABLE'}")
    print("=" * 70)
    print("🔧 MODEL STATUS:")
    print(f"  • COVID Audio Model: {'✅ LOADED' if covid_model else '❌ NOT AVAILABLE'}")
    print(f"  • Disease Model: {'✅ LOADED' if disease_model else '❌ NOT AVAILABLE'}")
    print(f"  • Disease Scaler: {'✅ LOADED' if disease_scaler else '❌ NOT AVAILABLE'}")
    print("=" * 70)
    print("📝 AVAILABLE ENDPOINTS:")
    print(f"  • Home: http://localhost:{port}/")
    print(f"  • Health Check: http://localhost:{port}/health")
    print(f"  • Test: http://localhost:{port}/test")
    print(f"  • Prediction: POST http://localhost:{port}/predict")
    print(f"  • Debug Models: http://localhost:{port}/api/debug/models")
    print("=" * 70)
    print("🚀 Starting server...")
    print("=" * 70)
    
    app.run(host="0.0.0.0", port=port, debug=True)