
import zipfile
import json
import os

MODEL_PATH = "backend/models/voiceguard_audio_model_final.keras"

if os.path.exists(MODEL_PATH):
    print(f"Inspecting {MODEL_PATH}...")
    try:
        with zipfile.ZipFile(MODEL_PATH, 'r') as zip_ref:
            # Keras 3 stores config in config.json
            if 'config.json' in zip_ref.namelist():
                with zip_ref.open('config.json') as f:
                    config = json.load(f)
                    print("✅ Found config.json")
                    print(f"Keras Version in file: {config.get('keras_version', 'Unknown')}")
                    print(f"Backend in file: {config.get('backend', 'Unknown')}")
            else:
                print("❌ config.json NOT found. This might not be a Keras 3 file.")
                
            # Check for other files
            print("Files in archive:", zip_ref.namelist())
            
    except Exception as e:
        print(f"❌ Error inspecting zip: {e}")
else:
    print(f"❌ File not found: {MODEL_PATH}")
