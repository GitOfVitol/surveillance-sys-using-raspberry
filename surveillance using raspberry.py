from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import numpy as np
import face_recognition
import pickle
import time
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import tensorflow as tf

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

import dropbox
import os

#face_detection에 사용할 모델
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile('./data/frozen_inference_graph_face.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        config = tf.compat.v1.ConfigProto()
    sess=tf.compat.v1.Session(graph=detection_graph, config=config)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
    classes_tensor = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections_tensor = detection_graph.get_tensor_by_name('num_detections:0')

def get_mobilenet_face(image):
    image = np.expand_dims(image, axis=0)
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes_tensor, scores_tensor, classes_tensor, num_detections_tensor],
        feed_dict={image_tensor: image})
    return (boxes, scores, classes, num_detections)

#dropbox 접속을 위한 토큰
dropbox_token = 'your token'
dbx = dropbox.Dropbox(dropbox_token)

#미리 추출한 임베딩값들로 된 pickle file load
encoding_file = './data/faceEncodings.pickle'
data = pickle.loads(open(encoding_file, "rb").read())

#firebase에서 생성한 database 접속을 위한 credit
cred = credentials.Certificate('your cred')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'your URL'
})

unknown_name = 'Unknown'
recognized_name = None
frame_count = 0
frame_interval = 8

frame_width = 640
frame_height = 480
frame_resolution = [frame_width, frame_height]
frame_rate = 16

#카메라 모듈 사용을 위한 설정
camera = PiCamera()
camera.resolution = frame_resolution
camera.framerate = frame_rate
rawCapture = PiRGBArray(camera, size=(frame_resolution))
time.sleep(0.1)

#캡쳐한 이미지 저장 위치
catured_image = './image/captured_image.jpg'

#카메라 모듈에서 frame capture하면서 수행할 작업들
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    start_time = time.time()
    image = frame.array
    camera.capture(catured_image)
    (height, width) = frame.shape[:2]

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    (boxes, scores, classes, num_detections) = get_mobilenet_face(rgb)

    rois = []
    for i, box in enumerate(boxes[0]) :
        if(scores[0][i]<0.6) : continue
        ymin, xmin, ymax, xmax = box
        (top, right, bottom, left) = (ymin * height, xmax * width, ymax * height, xmin * width)
        top, right, bottom, left = int(top), int(right), int(bottom), int(left)
        rois.append(tuple((top, right, bottom, left)))

    encodings = face_recognition.face_encodings(rgb, rois)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = unknown_name

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        names.append(name)

    for ((top, right, bottom, left), name) in zip(rois, names):
        y = top - 15 if top - 15 > 15 else top + 15
        color = (0, 255, 0)
        line = 2
        if(name == unknown_name):
            color = (0, 0, 255)
            line = 1
            
        if(name != recognized_name):
            recognized_name = name
            print("Send Notice")
            current = str(time.time())
            path = '/' + current[:10] + name + '.jpg'
            print(path)

            now = datetime.now()
            timeinfo = now.strftime('%Y년 %m월 %d일 %H시 %M분 %S초')
            
            ref = db.reference('surveillance')
            box_ref = ref.child(name)
            box_ref.update({
                'name': name,
                'time': timeinfo,
                'path': path
            })
            dbx.files_upload(open(catured_image, "rb").read(), path)
            print(dbx.files_get_metadata(path))

            if(recognized_name == unknown_name):
                s = smtplib.SMTP('smtp.gmail.com', 587)
                s.starttls()
                s.login('your email', 'your code')
                msgcontent = f"name : {name}\ntime : {timeinfo}\npath : {path}"
                msg = MIMEText(msgcontent)
                msg['Subject'] = 'Surveillance system - Unknown person is detected'
                s.sendmail("your email", "your email", msg.as_string())
                s.quit()

        cv2.rectangle(image, (left, top), (right, bottom), color, line)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, color, line)
                
    end_time = time.time()
    process_time = end_time - start_time
    print("=== A frame took {:.3f} seconds".format(process_time))

    cv2.imshow("출입자 검출 프로그램", image)

    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)

    if key == ord("q"):
        break
