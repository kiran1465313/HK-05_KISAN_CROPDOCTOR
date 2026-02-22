import cv2, numpy as np
from tflite_runtime.interpreter import Interpreter

MODEL = 'punjab_crop_final_model_strict_clean_best_acc.tflite'
interpreter = Interpreter(model_path=MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"✅ Model OK")
print(f"Input shape: {input_details[0]['shape']}")
print(f"Disease output: {output_details[0]['shape']}")
print(f"Severity output: {output_details[1]['shape']}")

# Handle 5D input [1,1,H,W,C]
input_shape = input_details[0]['shape']
if len(input_shape) == 5:
    h, w = input_shape[2], input_shape[3]
else:
    h, w = input_shape[1], input_shape[2]
print(f"Resize to: {h}x{w}")

cap = cv2.VideoCapture(0)
for i in range(10):
    ret, frame = cap.read()
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (w, h))
        inp = np.expand_dims(np.expand_dims(resized, 0), 0).astype(np.float32) / 255.0  # 5D
        
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        
        disease = interpreter.get_tensor(output_details[0]['index'])[0]
        severity = interpreter.get_tensor(output_details[1]['index'])[0]
        
        d_idx, d_conf = np.argmax(disease), np.max(disease)
        s_idx, s_conf = np.argmax(severity), np.max(severity)
        
        print(f"Frame {i}: Disease[{d_idx}]={d_conf:.3f}, Severity[{s_idx}]={s_conf:.3f}")
        
        cv2.putText(frame, f"D:{le.classes_[d_idx]}({d_conf:.1%})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow('DEMO', frame)
        if cv2.waitKey(500) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print("✅ TEST COMPLETE!")
