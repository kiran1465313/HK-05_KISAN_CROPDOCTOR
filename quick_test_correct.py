import cv2, numpy as np
from tflite_runtime.interpreter import Interpreter

MODEL = 'punjab_crop_final_model_strict_clean_best_acc.tflite'
interpreter = Interpreter(model_path=MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"✅ Input: {input_details[0]['shape']} = 4D [1,H,W,C]")
print(f"⚠️  Output 0 [Severity]: {output_details[0]['shape']} = 4 classes")
print(f"✅ Output 1 [Disease]: {output_details[1]['shape']} = 14 classes")

h, w = input_details[0]['shape'][1:3]  # Correct 4D slicing
print(f"Resize: {w}x{h}")

with open('disease_labels.txt', 'r') as f:
    disease_labels = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(0)
print("\n🧪 Testing LIVE detection (Press Q to quit)...")

while True:
    ret, frame = cap.read()
    if not ret: continue
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w, h))
    inp = np.expand_dims(resized, 0).astype(np.float32) / 255.0  # Correct 4D [1,H,W,C]
    
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    
    # CORRECTED: Output 0=Severity(4), Output 1=Disease(14)
    severity_out = interpreter.get_tensor(output_details[0]["index"])[0]  # [4]
    disease_out = interpreter.get_tensor(output_details[1]["index"])[0]   # [14]
    
    s_idx, s_conf = np.argmax(severity_out), np.max(severity_out)
    d_idx, d_conf = np.argmax(disease_out), np.max(disease_out)
    
    severity_name = ["healthy","mild","moderate","severe"][s_idx]
    
    print(f"🌱 Disease[{d_idx}]={d_conf:.3f}: {disease_labels[d_idx] if d_idx<len(disease_labels) else 'unknown'}")
    print(f"⚠️  Severity[{s_idx}]={s_conf:.3f}: {severity_name}")
    print(f"💧 Water: {WATER_BY_SEVERITY[severity_name.lower()]['ml']}ml\n")
    
    # Draw
    color = (0,255,0) if s_conf > 0.5 else (0,165,255)
    cv2.putText(frame, f"D:{disease_labels[d_idx][:20]}({d_conf:.0%})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"S:{severity_name}({s_conf:.0%})", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imshow('LIVE DEMO', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print("✅ DEMO COMPLETE!")
