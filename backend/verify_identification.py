
import os
import sys

# Add backend to path so we can import app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import rule_based_disease_detection, assess_severity
    print("✅ Successfully imported functions from app.py")
except ImportError as e:
    print(f"❌ Failed to import from app.py: {e}")
    sys.exit(1)

def run_test_case(name, symptoms, age, days):
    print(f"\n--- Testing: {name} ---")
    print(f"Inputs: Age={age}, Days={days}, Symptoms={symptoms}")
    
    # 1. Test Disease Detection
    disease, confidence, scores, reasoning = rule_based_disease_detection(symptoms, age, days)
    print(f"Resulting Disease: {disease} (Confidence: {confidence:.2f})")
    
    # 2. Test Severity Assessment
    severity_level, severity_text, severity_score, reasons = assess_severity(disease, symptoms, age, days)
    print(f"Resulting Severity: {severity_text} (Score: {severity_score}, Level: {severity_level})")
    
    return disease, severity_text

def main():
    test_cases = [
        # 1. Normal Cold
        {
            "name": "Normal Cold - Mild",
            "symptoms": {"fever": "yes", "dry_cough": "yes"},
            "age": 25, "days": 3,
            "expected_disease": "Normal_Cold", "expected_severity": "Mild"
        },
        {
            "name": "Normal Cold - Moderate",
            "symptoms": {"fever": "yes", "dry_cough": "yes"},
            "age": 25, "days": 8,
            "expected_disease": "Normal_Cold", "expected_severity": "Moderate"
        },
        {
            "name": "Normal Cold - Severe",
            "symptoms": {"fever": "yes", "dry_cough": "yes"},
            "age": 75, "days": 15,
            "expected_disease": "Normal_Cold", "expected_severity": "Severe"
        },
        # 2. COVID
        {
            "name": "COVID - Mild",
            "symptoms": {"smell": "yes"},
            "age": 30, "days": 3,
            "expected_disease": "COVID", "expected_severity": "Mild"
        },
        {
            "name": "COVID - Moderate",
            "symptoms": {"fever": "yes", "dry_cough": "yes", "fatigue": "yes"},
            "age": 40, "days": 8, 
            "expected_disease": "COVID", "expected_severity": "Moderate"
        },
        {
            "name": "COVID - Severe",
            "symptoms": {"breath": "yes", "fever": "yes", "dry_cough": "yes"},
            "age": 65, "days": 10,
            "expected_disease": "COVID", "expected_severity": "Severe"
        },
        # 3. Asthma
        {
            "name": "Asthma - Mild",
            "symptoms": {"wheezing": "yes"},
            "age": 20, "days": 2,
            "expected_disease": "Asthma", "expected_severity": "Mild"
        },
        {
            "name": "Asthma - Moderate",
            "symptoms": {"wheezing": "yes", "breath": "yes"},
            "age": 20, "days": 5,
            "expected_disease": "Asthma", "expected_severity": "Moderate"
        },
        {
            "name": "Asthma - Severe",
            "symptoms": {"wheezing": "yes", "breath": "yes", "chest": "yes"},
            "age": 20, "days": 5,
            "expected_disease": "Asthma", "expected_severity": "Severe"
        },
        # 4. Bronchitis
        {
            "name": "Bronchitis - Mild",
            "symptoms": {"wet_cough": "yes"},
            "age": 45, "days": 5,
            "expected_disease": "Bronchitis", "expected_severity": "Mild"
        },
        {
            "name": "Bronchitis - Moderate",
            "symptoms": {"wet_cough": "yes"},
            "age": 50, "days": 15,
            "expected_disease": "Bronchitis", "expected_severity": "Moderate"
        },
        {
            "name": "Bronchitis - Severe",
            "symptoms": {"wet_cough": "yes", "smoker": "yes"},
            "age": 55, "days": 22,
            "expected_disease": "Bronchitis", "expected_severity": "Severe"
        }
    ]
    
    success_count = 0
    for case in test_cases:
        actual_disease, actual_severity = run_test_case(
            case["name"], case["symptoms"], case["age"], case["days"]
        )
        
        disease_ok = actual_disease == case["expected_disease"]
        severity_ok = actual_severity == case["expected_severity"]
        
        if disease_ok and severity_ok:
            print("✅ PASS")
            success_count += 1
        else:
            if not disease_ok:
                print(f"❌ FAIL (Disease): Expected {case['expected_disease']}, got {actual_disease}")
            if not severity_ok:
                print(f"❌ FAIL (Severity): Expected {case['expected_severity']}, got {actual_severity}")
                
    print("\n" + "="*30)
    print(f"FINAL RESULTS: {success_count}/{len(test_cases)} tests passed")
    print("="*30)

if __name__ == "__main__":
    main()
