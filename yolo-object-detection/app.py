from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import time
import base64
import numpy as np
import cv2
from ultralytics import YOLO
import json

# Flask 앱 초기화
app = Flask(__name__, template_folder='./www', static_folder='./www', static_url_path='/')
CORS(app)  # CORS 설정 추가

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')

# 임시 이미지 저장 경로
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.json:
        return jsonify({'error': 'No image data provided'}), 400

    try:
        # 이미지 데이터 파싱
        img_data = request.json['image']
        # Base64 데이터에서 헤더 제거
        if ',' in img_data:
            img_data = img_data.split(',')[1]

        # Base64 디코딩
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 이미지 저장 (디버깅용)
        timestamp = int(time.time())
        img_path = f"{UPLOAD_FOLDER}/image_{timestamp}.jpg"
        cv2.imwrite(img_path, image)

        # YOLOv8로 객체 감지
        results = model(image)

        # 결과 처리
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = model.names[cls]

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': cls,
                    'name': name
                })

        return jsonify({
            'success': True,
            'detections': detections
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)