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

    try:
        # 업로드된 파일을 이미지로 변환
        input_image = Image.open(file).convert("RGB")
        print(f"Original image size: {input_image.size}")

        # 가로와 세로를 변경 (90도 회전)
        input_image = input_image.transpose(Image.ROTATE_270)
        print(f"Rotated image size: {input_image.size}")
    except Exception as e:
        return jsonify({'error': f'Invalid image format: {str(e)}'}), 400

    # YOLOv8 모델로 객체 탐지
    results = model(input_image)
    detections = results[0].boxes.data.cpu().numpy() 

    # 신뢰도
    confidence_threshold = 0.5
    
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font_path_korean = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

    try:
        font = ImageFont.truetype(font_path_korean, 40) 
    except IOError:
        print(f"Font not found at {font_path_korean}. Using default font.")
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
        main_color_rgb = get_main_color(cropped_box)
        main_color_name = rgb_to_color_name(main_color_rgb) 

        print(f"Detected object: {x_min}, {y_min}, {x_max}, {y_max}, confidence: {confidence}, class: {class_name}, main color: {main_color_name}")

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
        draw.rectangle([x_min, y_min, x_max, y_max], outline="black", width=10)

        # 클래스 이름과 신뢰도를 박스 위에 추가 (옵션)
        if confidence is not None and class_idx is not None:
            label = f"{main_color_name}"
                # 텍스트 경계 상자 계산
            bbox = draw.textbbox((0, 0), label, font=font)  # (0, 0)은 임시 위치
            text_width = bbox[2] - bbox[0]  # 너비 계산
            text_height = bbox[3] - bbox[1]  # 높이 계산

            # 텍스트의 최적 위치 계산 (텍스트가 박스 위에 표시되도록 조정)
            text_x = x_min  # 텍스트는 박스의 왼쪽 상단 x_min에서 시작
            text_y = max(0, y_min - text_height - 5)  # 텍스트가 이미지 경계를 벗어나지 않도록 보정

            text_position = (text_x, text_y)

            # 텍스트 추가
            draw.text(text_position, label, fill=main_color_rgb, font=font)

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
        
        # 이미지 크기 가져오기
    width, height = image.size

    # 중앙 부분 잘라내기 (가로 반, 세로 반)
    left = width // 4
    upper = height // 4
    right = 3 * (width // 4)
    lower = 3 * (height // 4)
    cropped_image = image.crop((left, upper, right, lower))

    # 크롭된 이미지를 50x50으로 축소
    cropped_image = cropped_image.resize((50, 50))
        
    # image = image.resize((50, 50))  
    pixels = np.array(cropped_image).reshape(-1, 3)  
    counter = Counter([tuple(pixel) for pixel in pixels])
    most_common_color = counter.most_common(1)[0][0]  
    return most_common_color




def rgb_to_color_name(rgb):
    """Convert RGB to a refined color name with range conditions."""
    r, g, b = rgb
    print(f"RGB: {r}, {g}, {b}")
    # 밝은 색 계열
    if 230 <= r <= 255 and 230 <= g <= 255 and 230 <= b <= 255:
        return "흰색"
    elif 230 <= r <= 255 and 0 <= g <= 100 and 0 <= b <= 100:
        return "밝은 빨간색"
    elif 230 <= r <= 255 and 100 < g <= 180 and 0 <= b <= 100:
        return "밝은 주황색"
    elif 230 <= r <= 255 and 180 < g <= 255 and 0 <= b <= 100:
        return "밝은 노란색"
    elif 0 <= r <= 100 and 230 <= g <= 255 and 0 <= b <= 100:
        return "밝은 초록색"
    elif 0 <= r <= 100 and 0 <= g <= 100 and 230 <= b <= 255:
        return "밝은 파랑색"
    elif 0 <= r <= 100 and 230 <= g <= 255 and 230 <= b <= 255:
        return "밝은 청록색"
    elif 230 <= r <= 255 and 0 <= g <= 100 and 230 <= b <= 255:
        return "밝은 보라색"
    elif 230 <= r <= 255 and 150 <= g <= 230 and 200 <= b <= 255:
        return "밝은 분홍색"

    # 중간 색 계열
    elif 150 <= r < 230 and 150 <= g < 230 and 150 <= b < 230:
        return "중간 회색"
    elif 150 <= r < 230 and 0 <= g <= 100 and 0 <= b <= 100:
        return "중간 빨간색"
    elif 150 <= r < 230 and 100 < g <= 180 and 0 <= b <= 100:
        return "중간 주황색"
    elif 150 <= r < 230 and 180 < g <= 230 and 0 <= b <= 100:
        return "중간 노란색"
    elif 0 <= r <= 100 and 150 <= g < 230 and 0 <= b <= 100:
        return "중간 초록색"
    elif 0 <= r <= 100 and 0 <= g <= 100 and 150 <= b < 230:
        return "중간 파랑색"
    elif 0 <= r <= 100 and 150 <= g < 230 and 150 <= b < 230:
        return "중간 청록색"
    elif 150 <= r < 230 and 0 <= g <= 100 and 150 <= b < 230:
        return "중간 보라색"
    elif 150 <= r < 230 and 150 <= g < 230 and 200 <= b < 230:
        return "중간 분홍색"

    # 어두운 색 계열
    elif 80 <= r < 150 and 80 <= g < 150 and 80 <= b < 150:
        return "어두운 회색"
    elif 80 <= r < 150 and 0 <= g <= 60 and 0 <= b <= 60:
        return "어두운 빨간색"
    elif 80 <= r < 150 and 60 < g <= 120 and 0 <= b <= 60:
        return "어두운 주황색"
    elif 80 <= r < 150 and 120 < g <= 150 and 0 <= b <= 60:
        return "어두운 노란색"
    elif 0 <= r <= 60 and 80 <= g < 150 and 0 <= b <= 60:
        return "어두운 초록색"
    elif 0 <= r <= 60 and 0 <= g <= 60 and 80 <= b < 150:
        return "어두운 파랑색"
    elif 0 <= r <= 60 and 80 <= g < 150 and 80 <= b < 150:
        return "어두운 청록색"
    elif 80 <= r < 150 and 0 <= g <= 60 and 80 <= b < 150:
        return "어두운 보라색"
    elif 80 <= r < 150 and 80 <= g < 150 and 100 <= b < 150:
        return "어두운 분홍색"

    # 특별 색상
    elif 0 <= r <= 50 and 0 <= g <= 50 and 0 <= b <= 50:
        return "검정색"
    elif 240 <= r <= 255 and 200 <= g <= 220 and 200 <= b <= 220:
        return "연핑크색"
    elif 180 <= r <= 200 and 120 <= g <= 180 and 60 <= b <= 120:
        return "갈색"

    # 기본값 (모든 RGB는 위 조건에 포함되므로 이 값이 호출되지 않음)
    else:
        return "기타 색상"


if __name__ == '__main__':
    app.run('0.0.0.0', port=8080, debug=True)