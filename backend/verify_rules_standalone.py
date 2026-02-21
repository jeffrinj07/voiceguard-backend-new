
# Standalone logic verification for VoiceGuard AI
# Rules copied directly from app.py at 2026-02-21

def rule_based_disease_detection(symptoms_dict, age, cough_days, audio_features=None):
    try:
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
        
        scores = {"Normal_Cold": 0, "COVID": 0, "Asthma": 0, "Bronchitis": 0}
        
        # NORMAL COLD
        cold_score = 0
        if (fever == 1 or sore_throat == 1) and (dry_cough == 1 or wet_cough == 1): cold_score += 30
        if breath == 0 and chest == 0 and wheezing == 0: cold_score += 20
        else: cold_score -= 15
        if smell == 0 and fatigue == 0: cold_score += 15
        if cough_days <= 7: cold_score += 15
        elif cough_days > 14: cold_score -= 10
        if age < 60: cold_score += 10
        else: cold_score -= 5
        scores["Normal_Cold"] = max(0, min(100, cold_score))
        
        # COVID
        covid_score = 0
        if smell == 1: covid_score += 40
        if fever == 1 and dry_cough == 1 and fatigue == 1: covid_score += 30
        if breath == 1: covid_score += 20
        if 5 <= cough_days <= 14: covid_score += 10
        elif cough_days > 14: covid_score += 5
        if age > 50: covid_score += 10
        scores["COVID"] = max(0, min(100, covid_score))
        
        # ASTHMA
        asthma_score = 0
        if wheezing == 1: asthma_score += 40
        if (breath == 1 or chest == 1) and fever == 0: asthma_score += 30
        if night_cough == 1: asthma_score += 20
        if is_chronic == 1: asthma_score += 15
        if wet_cough == 1: asthma_score -= 10
        if is_smoker == 1: asthma_score += 5
        if age < 40: asthma_score += 10
        scores["Asthma"] = max(0, min(100, asthma_score))
        
        # BRONCHITIS
        bronchitis_score = 0
        if wet_cough == 1: bronchitis_score += 35
        if cough_days > 14: bronchitis_score += 25
        if is_smoker == 1: bronchitis_score += 25
        if chest == 1: bronchitis_score += 15
        if is_chronic == 1: bronchitis_score += 15
        if wheezing == 1: bronchitis_score += 10
        if fever == 1: bronchitis_score += 10
        scores["Bronchitis"] = max(0, min(100, bronchitis_score))
        
        top_disease = max(scores, key=scores.get)
        confidence = scores[top_disease] / 100.0
        return top_disease, confidence, scores
    except Exception as e:
        return "Normal_Cold", 0.6, {}

def assess_severity(disease, symptoms_dict, age, cough_days, audio_features=None):
    try:
        fever = 1 if symptoms_dict.get('fever', 'no') == 'yes' else 0
        breath = 1 if symptoms_dict.get('breath', 'no') == 'yes' else 0
        chest = 1 if symptoms_dict.get('chest', 'no') == 'yes' else 0
        wheezing = 1 if symptoms_dict.get('wheezing', 'no') == 'yes' else 0
        fatigue = 1 if symptoms_dict.get('fatigue', 'no') == 'yes' else 0
        smell = 1 if symptoms_dict.get('smell', 'no') == 'yes' else 0
        
        severity_score = 0
        if cough_days > 14: severity_score += 30
        elif cough_days > 7: severity_score += 15
        if breath == 1: severity_score += 25
        if chest == 1: severity_score += 20
        if wheezing == 1: severity_score += 20
        if fever == 1:
            if age > 65 or age < 5: severity_score += 20
            else: severity_score += 10
        if fatigue == 1: severity_score += 5
        if age > 65: severity_score += 20
        elif age < 5: severity_score += 15
        elif age > 50: severity_score += 10

        if disease == "COVID":
            if smell == 1: severity_score += 5
            if breath == 1: severity_score += 15
        elif disease == "Asthma":
            if wheezing == 1 and breath == 1: severity_score += 20
        elif disease == "Bronchitis":
            is_smoker = 1 if symptoms_dict.get('smoker', 'no') == 'yes' else 0
            if is_smoker == 1 and cough_days > 21: severity_score += 15
            
        if severity_score >= 60: return 2, "Severe", severity_score
        elif severity_score >= 30: return 1, "Moderate", severity_score
        else: return 0, "Mild", severity_score
    except Exception as e:
        return 0, "Mild", 0

def run_test(name, symptoms, age, days, expected_d, expected_s):
    d, c, s_map = rule_based_disease_detection(symptoms, age, days)
    sl, st, ss = assess_severity(d, symptoms, age, days)
    
    d_ok = (d == expected_d)
    s_ok = (st == expected_s)
    
    status = "✅ PASS" if (d_ok and s_ok) else "❌ FAIL"
    print(f"[{status}] {name}")
    if not d_ok: print(f"    Expected Disease: {expected_d}, Got: {d} (Scores: {s_map})")
    if not s_ok: print(f"    Expected Severity: {expected_s}, Got: {st} (Score: {ss})")
    
    return (d_ok and s_ok)

def main():
    print("="*50)
    print("Verification of 12 Core Disease/Severity Scenarios")
    print("="*50)
    
    test_cases = [
        # Normal Cold
        ("Cold Mild", {"fever": "yes", "dry_cough": "yes"}, 25, 3, "Normal_Cold", "Mild"),
        ("Cold Moderate", {"fever": "yes", "dry_cough": "yes"}, 25, 9, "Normal_Cold", "Moderate"),
        ("Cold Severe", {"fever": "yes", "dry_cough": "yes", "breath": "yes"}, 70, 15, "Normal_Cold", "Severe"),
        # COVID
        ("COVID Mild", {"smell": "yes"}, 30, 2, "COVID", "Mild"),
        ("COVID Moderate", {"fever": "yes", "dry_cough": "yes", "fatigue": "yes"}, 55, 10, "COVID", "Moderate"),
        ("COVID Severe", {"fever": "yes", "dry_cough": "yes", "breath": "yes"}, 66, 10, "COVID", "Severe"),
        # Asthma
        ("Asthma Mild", {"wheezing": "yes"}, 20, 2, "Asthma", "Mild"),
        ("Asthma Moderate", {"wheezing": "yes", "breath": "yes"}, 20, 5, "Asthma", "Moderate"),
        ("Asthma Severe", {"wheezing": "yes", "breath": "yes", "chest": "yes"}, 20, 7, "Asthma", "Severe"),
        # Bronchitis
        ("Bronchitis Mild", {"wet_cough": "yes"}, 40, 5, "Bronchitis", "Mild"),
        ("Bronchitis Moderate", {"wet_cough": "yes"}, 40, 16, "Bronchitis", "Moderate"),
        ("Bronchitis Severe", {"wet_cough": "yes", "smoker": "yes", "breath": "yes"}, 55, 22, "Bronchitis", "Severe")
    ]
    
    passed = 0
    for tc in test_cases:
        if run_test(*tc): passed += 1
        
    print("="*50)
    print(f"Summary: {passed}/{len(test_cases)} Passed")
    print("="*50)

if __name__ == "__main__":
    main()
