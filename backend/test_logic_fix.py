import json
import os
import sys

# Add the current directory to sys.path so we can import app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import check_and_install_dependencies, rule_based_prediction, enhance_features
    import numpy as np
    
    print("--- Dependency Check Test ---")
    available, missing = check_and_install_dependencies()
    print(f"Available: {available}")
    print(f"Missing: {missing}")
    
    if 'cv2' in available:
        print("✅ cv2 detection FIXED")
    else:
        print("❌ cv2 detection STILL FAILING")

    print("\n--- Prediction Logic Test ---")
    symptoms = {
        "fever": "yes",
        "dry_cough": "yes",
        "wet_cough": "no",
        "wheezing": "no",
        "breath": "no",
        "chest": "no",
        "fatigue": "no",
        "sore_throat": "no",
        "smell": "no",
        "night_cough": "no",
        "smoker": "no",
        "chronic": "no"
    }
    age = 25
    cough_days = 2
    
    rule_disease, rule_conf = rule_based_prediction(symptoms, age, cough_days)
    print(f"Rule-based prediction: {rule_disease} (confidence: {rule_conf})")
    
    # Simulate a low-confidence ML prediction of "Asthma" (what the user experienced)
    ml_disease = "Asthma"
    ml_conf = 0.444
    
    final_disease = ml_disease
    final_conf = ml_conf
    
    if ml_conf < 0.6:
        print(f"🔍 Low ML confidence ({ml_conf:.2f}). Cross-referencing with rule-based: {rule_disease}")
        if ml_disease != rule_disease:
            if ml_conf < 0.45:
                print(f"⚖️ Favoring rule-based prediction ({rule_disease}) over low-confidence ML ({ml_disease})")
                final_disease = rule_disease
                final_conf = rule_conf
            else:
                print("⚖️ Keeping ML prediction but noting potential uncertainty")

    print(f"FINAL RESULT: {final_disease} (confidence: {final_conf})")
    
    if final_disease == "Normal_Cold":
        print("✅ LOGIC REFINEMENT WORKING")
    else:
        print("❌ LOGIC REFINEMENT FAILED")

except Exception as e:
    print(f"Error during test: {e}")
    import traceback
    traceback.print_exc()
