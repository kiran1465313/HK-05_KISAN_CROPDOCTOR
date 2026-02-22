import cv2, numpy as np
from tflite_runtime.interpreter import Interpreter
import time

MODEL = 'punjab_crop_final_model_strict_clean_best_acc.tflite'
interpreter = Interpreter(model_path=MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Model: {MODEL}")
print(f"Input: {input_details[0]['shape']}")
print(f"Disease output: {output_details[0]['shape']}")
print(f"Severity output: {output_details[1]['shape']}")

cap = cv2.VideoCapture(0)
print("Testing 5 frames...")

for i in range(5):
    ret, frame = cap.read()
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = input_details[0]['shape'][1:]
        resized = cv2.resize(rgb, (w, h))
        inp = np.expand_dims(resized, 0).astype(np.float32) / 255.0
        
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        
        disease = interpreter.get_tensor(output_details[0]['index'])[0]
        severity = interpreter.get_tensor(output_details[1]['index'])[0]
        
        d_idx, d_conf = np.argmax(disease), np.max(disease)
        s_idx = np.argmax(severity)
        
        print(f"Frame {i}: Disease[{d_idx}]={d_conf:.3f}, Severity[{s_idx}]")
        cv2.imshow('test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
