def rule_based_prediction(symptoms_dict, age, cough_days):
    fever = 1 if symptoms_dict.get('fever', 'no') == 'yes' else 0
    dry_cough = 1 if symptoms_dict.get('dry_cough', 'no') == 'yes' else 0
    sore_throat = 1 if symptoms_dict.get('sore_throat', 'no') == 'yes' else 0
    
    if (fever == 1 or sore_throat == 1) and (dry_cough == 1):
        return "Normal_Cold", 0.85
    return "Normal_Cold", 0.65

print("--- Minimal Logic Test ---")
symptoms = {"fever": "yes", "dry_cough": "yes"}
rule_disease, rule_conf = rule_based_prediction(symptoms, 25, 2)

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

print(f"Final Result: {final_disease} (confidence: {final_conf})")
if final_disease == "Normal_Cold":
    print("✅ LOGIC REFINEMENT WORKING")
else:
    print("❌ LOGIC REFINEMENT FAILED")
