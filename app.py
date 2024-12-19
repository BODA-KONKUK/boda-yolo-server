import os  
from flask import Flask, request, jsonify
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO
from collections import Counter
import boto3
from io import BytesIO
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

app = Flask(__name__)

model = YOLO('yolov8n.pt')

# 환경 변수에서 AWS S3 설정 가져오기
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# AWS S3 설정
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

S3_BUCKET_NAME = 'boda-bucket'  

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    input_image = Image.open(file)
    print(f"Uploaded image size: {input_image.size}")

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
    
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    try:
        font = ImageFont.truetype(font_path, 40) 
    except IOError:
        print(f"Font not found at {font_path}. Using default font.")
        font = ImageFont.load_default()

    
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
        draw.rectangle([x_min, y_min, x_max, y_max], outline=main_color, width=20)

        # 클래스 이름과 신뢰도를 박스 위에 추가 (옵션)
        if confidence is not None and class_idx is not None:
            label = f"{class_name}"
            text_position = (x_min, y_min - 10)  # 텍스트 위치
            draw.text(text_position, label, fill=main_color,font=font)

    try:
        s3_url = save_image_to_s3(input_image, 'detected_image.png')
        print(f"Image uploaded to S3: {s3_url}")
    except Exception as e:
        return jsonify({'error': f'Failed to upload image to S3: {str(e)}'}), 500


    # 결과 이미지를 저장
    output_image_path = "./detected_image1.png"  # 저장할 이미지 경로
    input_image.save(output_image_path)
    print(f"Image saved with detections: {output_image_path}")

    response = {
        'status': 'success',
        'image':s3_url
        # 'total_detections': len(detected_objects)
    }
    print(response);

    return jsonify(response), 200

def save_image_to_s3(image, filename):
    """
    이미지 파일을 S3에 업로드하는 함수
    """
    buffer = BytesIO()
    image.save(buffer, format='PNG')  # 이미지를 버퍼에 저장
    buffer.seek(0)

    s3_client.upload_fileobj(
        buffer,
        S3_BUCKET_NAME,
        filename,
        ExtraArgs={'ContentType': 'image/png'}
    )

    s3_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{filename}"
    return s3_url


def get_main_color(image, num_colors=1):
    """Extract the main color from an image."""
    
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    image = image.resize((50, 50))  
    pixels = np.array(image).reshape(-1, 3)  
    counter = Counter([tuple(pixel) for pixel in pixels])
    most_common_color = counter.most_common(1)[0][0]  
    return most_common_color


if __name__ == '__main__':
    app.run('0.0.0.0', port=8080, debug=True)