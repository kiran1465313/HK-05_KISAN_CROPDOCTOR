from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import threading
import time
from datetime import datetime

app = Flask(__name__)

# ===== FIXED MODEL CONFIG =====
MODEL_PATH = "punjab_crop_final_model_strict_clean_best_acc.tflite"
LABELS_PATH = "disease_labels.txt"
MIN_CONF = 0.2  # Low for demo
PORT = 8080

WATER_BY_SEVERITY = {
    "healthy": {"ml": 0, "color": "#10b981", "emoji": "✅"},
    "mild": {"ml": 20, "color": "#f59e0b", "emoji": "⚠️"},
    "moderate": {"ml": 50, "color": "#ef4444", "emoji": "🚨"},
    "severe": {"ml": 50, "color": "#dc2626", "emoji": "🚨"}
}

severity_names = ["healthy", "mild", "moderate", "severe"]

# State
detection_state = {
    "disease": "Monitoring...",
    "severity": "healthy", 
    "water_ml": 0,
    "confidence": 0,
    "timestamp": "",
    "status": "ready"
}

history = []

# Load model
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

h, w = input_details[0]['shape'][1:3]  # FIXED 4D [1,H,W,C]
print(f"✅ Model: {MODEL_PATH}")
print(f"📐 Input: {input_details[0]['shape']} -> Resize {w}x{h}")
print(f"⚠️ Severity: {output_details[0]['shape']} (output 0)")
print(f"🌱 Disease: {output_details[1]['shape']} (output 1)")

with open(LABELS_PATH, "r") as f:
    disease_labels = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def generate_frames():
    global detection_state
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (w, h))
        input_data = np.expand_dims(resized, 0).astype(np.float32) / 255.0  # FIXED 4D

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        # FIXED: Output 0=Severity[4], Output 1=Disease[14]
        severity_output = interpreter.get_tensor(output_details[0]["index"])[0]
        disease_output = interpreter.get_tensor(output_details[1]["index"])[0]
        
        s_idx = np.argmax(severity_output)
        d_idx = np.argmax(disease_output)
        conf = max(severity_output[s_idx], disease_output[d_idx])
        
        if conf > MIN_CONF:
            disease_name = disease_labels[d_idx] if d_idx < len(disease_labels) else f"class_{d_idx}"
            severity_name = severity_names[s_idx]
            water_ml = WATER_BY_SEVERITY.get(severity_name.lower(), {"ml": 0})["ml"]
            
            detection_state.update({
                "disease": disease_name,
                "severity": severity_name,
                "water_ml": water_ml,
                "confidence": float(conf),
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "status": "detected"
            })
            
            history.append(detection_state.copy())
            if len(history) > 10: history.pop(0)

        # Beautiful overlay
        severity_info = WATER_BY_SEVERITY.get(detection_state["severity"].lower(), {"color": "#6b7280"})
        overlay_color = severity_info["color"]
        
        cv2.rectangle(frame, (30, 30), (frame.shape[1]-50, 200), (20, 20, 40), -1)
        cv2.putText(frame, f"🌱 DISEASE: {detection_state['disease'][:28]}", (50, 70), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"⚠️ SEVERITY: {detection_state['severity'].upper()}", (50, 105), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,255,255), 2)
        cv2.putText(frame, f"💧 WATER: {detection_state['water_ml']}ml", (50, 140), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,255,255), 2)
        cv2.putText(frame, f"🎯 CONF: {detection_state['confidence']:.0%}", (50, 175), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    return jsonify(detection_state)

@app.route('/api/history')
def api_history():
    return jsonify(history)

if __name__ == '__main__':
    print("🌟 PUNJAB CROP DOCTOR - SIH 2026")
    print(f"🌐 LIVE: http://0.0.0.0:{PORT}")
    print("📱 Phone: http://PI_IP:8080")
    print("🔗 ngrok: ngrok http 8080")
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
