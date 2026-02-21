
import requests
import json
import os

def test_predict():
    url = "http://localhost:5000/predict"
    
    # 1. Test "Alarmist" case (Fever + Dry Cough) - Should be Normal Cold now
    print("\nTest 1: Fever + Dry Cough (expecting Normal_Cold)")
    data = {"symptoms": json.dumps({"fever": "yes", "dry_cough": "yes", "fatigue": "no", "breath": "no"}),
            "age": 30,
            "cough_days": 10}
            
    try:
        response = requests.post(url, data=data)
        if response.status_code == 200:
            print(f"✅ Success: {response.json().get('disease')}")
        else:
            print(f"❌ Failed: {response.text}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")

    # 2. Test with Audio
    print("\nTest 2: With Audio File")
    audio_path = os.path.join(os.getcwd(), 'backend', 'test_audio.wav')
    if os.path.exists(audio_path):
        with open(audio_path, 'rb') as f:
            files = {'audio': f}
            try:
                response = requests.post(url, data=data, files=files)
                if response.status_code == 200:
                    res = response.json()
                    print(f"✅ Success: {res.get('disease')}")
                    print(f"ℹ️ Analysis: {res.get('covid_analysis_en')}")
                else:
                    print(f"❌ Failed: {response.text}")
            except Exception as e:
                print(f"❌ Connection Error: {e}")
    else:
        print("❌ test_audio.wav not found")

if __name__ == "__main__":
    test_predict()
