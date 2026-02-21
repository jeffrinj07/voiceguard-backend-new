
import sys
from unittest.mock import MagicMock

# Mock heavy/problematic dependencies before importing app
sys.modules['tensorflow'] = MagicMock()
sys.modules['keras'] = MagicMock()
sys.modules['librosa'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['firebase_admin'] = MagicMock()
sys.modules['google.cloud'] = MagicMock()
sys.modules['joblib'] = MagicMock()

import os
# Ensure dummy environment variables
os.environ['FIREBASE_CREDENTIALS'] = '{}'

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import rule_based_disease_detection, assess_severity
    print("✅ Successfully imported functions from app.py (with mocks)")
except Exception as e:
    print(f"❌ Failed to import from app.py: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def run_test_case(name, symptoms, age, days):
    print(f"\n--- Testing: {name} ---")
    
    # 1. Test Disease Detection
    disease, confidence, scores, reasoning = rule_based_disease_detection(symptoms, age, days)
    print(f"Resulting Disease: {disease} (Confidence: {confidence:.2f})")
    
    # 2. Test Severity Assessment
    severity_level, severity_text, severity_score, reasons = assess_severity(disease, symptoms, age, days)
    print(f"Resulting Severity: {severity_text} (Score: {severity_score})")
    
    return disease, severity_text

def main():
    # We want to verify 4 diseases x 3 severities
    # Diseases: Normal_Cold, COVID, Asthma, Bronchitis
    # Severity thresholds: Mild (<30), Moderate (30-59), Severe (>=60)
    
    test_cases = [
        # --- NORMAL COLD ---
        {
            "name": "Normal Cold - Mild",
            "symptoms": {"fever": "yes", "dry_cough": "yes"},
            "age": 25, "days": 3,
            "expected_disease": "Normal_Cold", "expected_severity": "Mild"
        },
        {
            "name": "Normal Cold - Moderate",
            "symptoms": {"fever": "yes", "dry_cough": "yes", "sore_throat": "yes"},
            "age": 25, "days": 8, # Days > 7 adds 15 to severity
            "expected_disease": "Normal_Cold", "expected_severity": "Moderate"
        },
        {
            "name": "Normal Cold - Severe",
            "symptoms": {"fever": "yes", "dry_cough": "yes", "breath": "yes"}, # Breath +25, fever +10, age +20 = 55 (Still moderate?)
            "age": 70, "days": 15, # days > 14 +30, age +20, fever +10 = 60 (Severe)
            "expected_disease": "Normal_Cold", "expected_severity": "Severe"
        },
        # --- COVID ---
        {
            "name": "COVID - Mild",
            "symptoms": {"smell": "yes"},
            "age": 30, "days": 3,
            "expected_disease": "COVID", "expected_severity": "Mild"
        },
        {
            "name": "COVID - Moderate",
            "symptoms": {"fever": "yes", "dry_cough": "yes", "fatigue": "yes"},
            "age": 40, "days": 8, # Severity: days>7 (+15), fever (+10) = 25 (Still mild?) 
            # Let's adjust for Moderate: age 55 (+10), days 8 (+15), fever (+10) = 35 (Moderate)
            "expected_disease": "COVID", "expected_severity": "Moderate"
        },
        {
            "name": "COVID - Severe",
            "symptoms": {"breath": "yes", "fever": "yes", "dry_cough": "yes"}, 
            # Severity: days>7 (+15), breath (+25), fever (+10), age 66 (+20) = 70 (Severe)
            "age": 66, "days": 10,
            "expected_disease": "COVID", "expected_severity": "Severe"
        },
        # --- ASTHMA ---
        {
            "name": "Asthma - Mild",
            "symptoms": {"wheezing": "yes"},
            "age": 20, "days": 2,
            "expected_disease": "Asthma", "expected_severity": "Mild"
        },
        {
            "name": "Asthma - Moderate",
            "symptoms": {"wheezing": "yes", "breath": "yes"}, # wheezing+20, breath+25 = 45 (Moderate)
            "age": 20, "days": 5,
            "expected_disease": "Asthma", "expected_severity": "Moderate"
        },
        {
            "name": "Asthma - Severe",
            "symptoms": {"wheezing": "yes", "breath": "yes", "chest": "yes"}, 
            # wheezing+20, breath+25, chest+20 = 65 (Severe)
            "age": 20, "days": 5,
            "expected_disease": "Asthma", "expected_severity": "Severe"
        },
        # --- BRONCHITIS ---
        {
            "name": "Bronchitis - Mild",
            "symptoms": {"wet_cough": "yes"},
            "age": 45, "days": 5,
            "expected_disease": "Bronchitis", "expected_severity": "Mild"
        },
        {
            "name": "Bronchitis - Moderate",
            "symptoms": {"wet_cough": "yes"}, # days>14 (+30) = 30 (Moderate)
            "age": 50, "days": 16,
            "expected_disease": "Bronchitis", "expected_severity": "Moderate"
        },
        {
            "name": "Bronchitis - Severe",
            "symptoms": {"wet_cough": "yes", "smoker": "yes", "breath": "yes"}, 
            # days>21 (+30), smoker+prolonged(+15), breath(+25) = 70 (Severe)
            "age": 55, "days": 22,
            "expected_disease": "Bronchitis", "expected_severity": "Severe"
        }
    ]
    
    # Special adjustment for COVID Moderate case in the test input
    test_cases[4]["age"] = 55 
    
    print("\n" + "="*50)
    print("🚀 STARTING VERIFICATION OF 12 CORE SCENARIOS")
    print("="*50)
    
    passed = 0
    for case in test_cases:
        actual_disease, actual_severity = run_test_case(case['name'], case['symptoms'], case['age'], case['days'])
        
        d_ok = (actual_disease == case['expected_disease'])
        s_ok = (actual_severity == case['expected_severity'])
        
        if d_ok and s_ok:
            print("✅ PASS")
            passed += 1
        else:
            if not d_ok: print(f"❌ DISEASE MISMATCH: Expected {case['expected_disease']}, got {actual_disease}")
            if not s_ok: print(f"❌ SEVERITY MISMATCH: Expected {case['expected_severity']}, got {actual_severity}")

    print("\n" + "="*50)
    print(f"📊 SUMMARY: {passed}/{len(test_cases)} cases passed")
    print("="*50)

if __name__ == "__main__":
    main()
