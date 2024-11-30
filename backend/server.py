from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.models import load_model
import os
import base64, ssl, json

app = Flask(__name__)

# model =load_model('path/to/load', compile=False) # <--------------------------------- 여기서 모델 로드

@app.route('/')
def index():
    return jsonify({'message': 'Flask server is running!'})


def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def build_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response
    
@app.route('/get_status', methods=['GET'])
def get_status():
    return jsonify({'status': '50%'})

@app.route('/upload', methods=['POST', 'OPTIONS'])
def process_video():
    if request.method == 'OPTIONS': 
        return build_preflight_response()
        
    elif request.method == 'POST': 
        data=json.loads(request.data)
        if 'video' not in data and 'image' not in data:
            response = app.response_class(
            response=json.dumps({'error': 'No video or image file uploaded'}),
            status=503,
            mimetype='application/json',
            )
            return response
    
        results=dict()
        
        if 'video' in data:
            print(type(data['video']))
            video_data = base64.b64decode(data['video'].split('base64,')[1])
            
            # video_data = data['video'].split('base64,')[1]
            # # Encode the video file as Base64
            # # video_data = base64.b64encode(video_file.read())

            # # Decode the Base64 string back to video file
            # decoded_video_data = base64.b64decode(video_data)

            # video_path = 'temp_video.mp4'
            # with open(video_path, "wb") as file:
            #     file.write(decoded_video_data)
            
            # cap = cv2.VideoCapture(video_path)
            # fps = cap.get(cv2.CAP_PROP_FPS)
            # if not cap.isOpened():
            #     os.remove(video_path)  # Remove the temporary video file
            #     response = app.response_class(
            #     response=json.dumps({'error': 'Failed to open video file'}),
            #     status=503,
            #     mimetype='application/json',
            #     )
            #     return response
            
            # while cap.isOpened():
            #     ret, frame = cap.read()
            #     if not ret:
            #         break
                
            #     img = tf.image.resize_with_pad(frame, 512, 512)
            #     model_out = np.array(model(np.array([img]))[0]) # <------------------------------------------------------- 여기서 후처리
            
            # results['video_result']=base64.b64encode(model_out.tobytes()).decode('UTF-8')
            results['video_result']=data['video']
            
        else:
            frame = base64.b64decode(data['image'].split('base64,')[1])
            print("\nchecking\n")
            print(type(frame))
            
            np_frame = np.frombuffer(frame, dtype=np.uint8)
            print(type(np_frame))
            
            img = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)  # BGR로 디코딩
            if img is None:
                print("img is NOOOOONE")
            else:
                print(f"image shape: {img.shape}")
                print(f"Type : {type(img)}")
            print("done\n")
            
            # img = tf.image.resize_with_pad(frame, 512, 512)
            img = tf.image.resize_with_pad(img, 512, 512)
            
            # model_out = np.array(model(np.array([img]))[0]) # <------------------------------------------------------- 여기서 후처리
            
            # results['image_result']=base64.b64encode(model_out.tobytes()).decode('UTF-8')
            results['image_result']=base64.b64encode(frame).decode('UTF-8')
            
            
        response = app.response_class(
        # response=json.dumps({'message': 'image 잘 받았어요'}),
        response=json.dumps(results),
        status=200,
        mimetype='application/json',
        )

        return response



# SSL=ssl.SSLContext(ssl.PROTOCOL_TLS)
# SSL.load_cert_chain(certfile='cert.pem', keyfile='key.pem')
CORS(app, resources={r"*": {"origins": "*"}})
# app.run(host='0.0.0.0', port=4000, ssl_context=SSL, debug=True)
app.run(host='0.0.0.0', port=4000, debug=True)
