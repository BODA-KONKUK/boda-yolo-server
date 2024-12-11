from flask import Flask, request, jsonify
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO
from collections import Counter

app = Flask(__name__)

model = YOLO('yolov8n.pt')

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    input_image = Image.open(file)

    try:
        # 업로드된 파일을 이미지로 변환
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({'error': f'Invalid image format: {str(e)}'}), 400

    # YOLOv8 모델로 객체 탐지
    results = model(image)
    detections = results[0].boxes.data.cpu().numpy() 

    # 신뢰도
    confidence_threshold = 0.5
    
    # font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"  
    # font = ImageFont.truetype(font_path, 20)
    
    # 결과를 JSON 형식으로 변환
    detected_objects = []
    for detection in detections:
        # numpy 배열의 값을 직접 사용
        x_min, y_min, x_max, y_max = detection[:4]
        confidence = detection[4] if len(detection) > 4 else None
        class_idx = detection[5] if len(detection) > 5 else None
        
        class_name = model.names.get(int(class_idx), "Unknown")
        
        print(f"Detected object: {x_min}, {y_min}, {x_max}, {y_max}, confidence: {confidence}, class: {class_name}")
        
        if confidence is not None and confidence < confidence_threshold:
            continue
        
        class_name = model.names.get(int(class_idx), "Unknown")
        
        # 박스 내부 이미지 추출
        cropped_box = input_image.crop((x_min, y_min, x_max, y_max))
        # 주요 색상 추출
        main_color = get_main_color(cropped_box)

        print(f"Detected object: {x_min}, {y_min}, {x_max}, {y_max}, confidence: {confidence}, class: {class_name}, main color: {main_color}")

        detected_object = {
            'bounding_box': {
                'x_min': int(x_min),
                'y_min': int(y_min),
                'x_max': int(x_max),
                'y_max': int(y_max)
            }
        }
        if confidence is not None:
            detected_object['confidence'] = round(float(confidence), 2)
        if class_idx is not None:
            detected_object['class_id'] = int(class_idx)
            detected_object['class_name'] = class_name


        detected_objects.append(detected_object)
        # 이미지 위에 박스 그리기
        draw = ImageDraw.Draw(input_image)
        draw.rectangle([x_min, y_min, x_max, y_max], outline=main_color, width=10)

        # 클래스 이름과 신뢰도를 박스 위에 추가 (옵션)
        # if confidence is not None and class_idx is not None:
        #     label = f"{class_name} {confidence:.2f}"
        #     text_position = (x_min, y_min - 10)  # 텍스트 위치
        #     draw.text(text_position, label, fill="red",font=font)

    # 결과 이미지를 저장
    output_image_path = "./detected_image1.png"  # 저장할 이미지 경로
    input_image.save(output_image_path)
    print(f"Image saved with detections: {output_image_path}")

    response = {
        'status': 'success',
        'detections': detected_objects,
        'total_detections': len(detected_objects)
    }


    return jsonify(response), 200



def get_main_color(image, num_colors=1):
    """Extract the main color from an image."""
    image = image.resize((50, 50))  
    pixels = np.array(image).reshape(-1, 3)  
    counter = Counter([tuple(pixel) for pixel in pixels])
    most_common_color = counter.most_common(1)[0][0]  
    return most_common_color


if __name__ == '__main__':
    app.run('0.0.0.0', port=8080, debug=True)