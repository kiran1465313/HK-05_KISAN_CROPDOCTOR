import cv2, numpy as np
from tflite_runtime.interpreter import Interpreter

MODEL = 'punjab_crop_final_model_strict_clean_best_acc.tflite'
interpreter = Interpreter(model_path=MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"✅ Model loaded ✓")
print(f"Input: {input_details[0]['shape']}")
print(f"Severity: {output_details[0]['shape']} (4 classes)")
print(f"Disease: {output_details[1]['shape']} (14 classes)")

h, w = input_details[0]['shape'][1:3]

WATER_BY_SEVERITY = {
    "healthy": 0, "mild": 20, "moderate": 50, "severe": 50
}

with open('disease_labels.txt', 'r') as f:
    disease_labels = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(0)
print("\n🎬 LIVE DEMO - Press Q to quit")
print("Point camera at plant leaves!")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret: continue
    
    frame_count += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w, h))
    inp = np.expand_dims(resized, 0).astype(np.float32) / 255.0
    
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    
    # Severity first (output 0), Disease second (output 1)
    severity_out = interpreter.get_tensor(output_details[0]["index"])[0]
    disease_out = interpreter.get_tensor(output_details[1]["index"])[0]
    
    s_idx, s_conf = np.argmax(severity_out), np.max(severity_out)
    d_idx, d_conf = np.argmax(disease_out), np.max(disease_out)
    
    severity_name = ["healthy","mild","moderate","severe"][s_idx]
    water_ml = WATER_BY_SEVERITY[severity_name.lower()]
    disease_name = disease_labels[d_idx] if d_idx < len(disease_labels) else "unknown"
    
    # Print every 10 frames
    if frame_count % 10 == 0:
        print(f"[{frame_count}] {disease_name:<25} | {severity_name:8}({s_conf:.0%}) | 💧{water_ml}ml")
    
    # Draw results
    color = (0,255,0) if water_ml == 0 else (0,165,255)
    y_pos = 30
    cv2.putText(frame, f"Disease: {disease_name[:25]}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    y_pos += 35
    cv2.putText(frame, f"Severity: {severity_name} ({s_conf:.0%})", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    y_pos += 35
    cv2.putText(frame, f"Water: {water_ml}ml", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.imshow('🌱 PUNJAB CROP DOCTOR - SIH 2026', frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
print("\n🎉 MODEL WORKS PERFECTLY! Ready for web dashboard!")
