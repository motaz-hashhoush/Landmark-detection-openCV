from flask import Flask, render_template, Response, request
import cv2
import datetime
import time
import os
import sys
import numpy as np
from threading import Thread
from PIL import Image
import matplotlib.pyplot as plt
import fld_utils
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

global capture, rec_frame, filter, switch, neg, face, rec, out
capture = 0
filter = 0
neg = 0
face = 0
switch = 1
rec = 0

# make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

# Load pretrained face detection model
net = cv2.dnn.readNetFromCaffe('./saved_models/deploy.prototxt.txt',
                               './saved_models/res10_300x300_ssd_iter_140000.caffemodel')

# instatiate flask app
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)


def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)


def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:
        return frame

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame = frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = (int(w * r), 480)
        frame = cv2.resize(frame, dim)
    except Exception as e:
        pass
    return frame


def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1],
                        1), dtype=overlay.dtype) * 255
            ],
            axis=2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * \
        background[y:y+h, x:x+w] + mask * overlay_image

    return background


def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame
    while True:
        success, frame = camera.read()
        if success:
            if(face):
                frame = detect_face(frame)
            if(filter):

                # Load original image
                face_img_path = frame
                orig_img = cv2.cvtColor(face_img_path, cv2.COLOR_BGR2RGB)
                orig_size_x, orig_size_y = orig_img.shape[0], orig_img.shape[1]

                # Prepare input image
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, dsize=(96, 96),
                                 interpolation=cv2.INTER_AREA)
                img = np.expand_dims(img, axis=2)
                img = img / 255
                img = img.astype("float32")

                # Predict landmarks
                model = fld_utils.load_model("cnn")
                y_pred = model.predict(np.expand_dims(img, axis=0))[0]
                landmarks = fld_utils.extract_landmarks(
                    y_pred, orig_size_x, orig_size_y)

                # Save original image with landmarks on top
               # fld_utils.save_img_with_landmarks(
                #    orig_img, landmarks, "test_img_prediction.png")

                # Extract x and y values from landmarks of interest
                left_eye_center_x = int(landmarks[0][0])
                left_eye_center_y = int(landmarks[0][1])
                right_eye_center_x = int(landmarks[1][0])
                right_eye_center_y = int(landmarks[1][1])
                left_eye_outer_x = int(landmarks[3][0])
                right_eye_outer_x = int(landmarks[5][0])

                # Load images using PIL
                # PIL has better functions for rotating and pasting compared to cv2

                sunglasses_img = Image.open(
                    "input/sunglasses.png")

                # Resize sunglasses
                sunglasses_width = int(
                    (left_eye_outer_x - right_eye_outer_x) * 1.4)
                sunglasses_height = int(
                    sunglasses_img.size[1] * (sunglasses_width / sunglasses_img.size[0]))
                sunglasses_resized = sunglasses_img.resize(
                    (sunglasses_width, sunglasses_height))

                # Rotate sunglasses
                eye_angle_radians = np.arctan(
                    (right_eye_center_y - left_eye_center_y) / (left_eye_center_x - right_eye_center_x))
                sunglasses_rotated = sunglasses_resized.rotate(np.degrees(
                    eye_angle_radians), expand=True, resample=Image.BICUBIC)

                # Compute positions such that the center of the sunglasses is
                # positioned at the center point between the eyes
                x_offset = int(sunglasses_width * 0.5)
                y_offset = int(sunglasses_height * 0.5)
                pos_x = int(
                    (left_eye_center_x + right_eye_center_x) / 2) - x_offset
                pos_y = int(
                    (left_eye_center_y + right_eye_center_y) / 2) - y_offset

                img = overlay_transparent(frame, np.array(
                    sunglasses_rotated), pos_x, pos_y)

                cv2.imshow('Glasses over face', frame)
                # Paste sunglasses on face image

            if(neg):
                frame = cv2.bitwise_not(frame)
            if(capture):
                capture = 0
                now = datetime.datetime.now()
                p = os.path.sep.join(
                    ['shots', "shot_{}.png".format(str(now).replace(":", ''))])
                cv2.imwrite(p, frame)

            if(rec):
                rec_frame = frame
                frame = cv2.putText(cv2.flip(
                    frame, 1), "Recording...", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                frame = cv2.flip(frame, 1)

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

        else:
            pass


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
        elif request.form.get('filter') == 'FILTER':
            global filter
            filter = not filter
        elif request.form.get('neg') == 'Negative':
            global neg
            neg = not neg
        elif request.form.get('face') == 'Face Only':
            global face
            face = not face
            if(face):
                time.sleep(4)
        elif request.form.get('stop') == 'Stop/Start':

            if(switch == 1):
                switch = 0
                camera.release()
                cv2.destroyAllWindows()

            else:
                camera = cv2.VideoCapture(0)
                switch = 1
        elif request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec = not rec
            if(rec):
                now = datetime.datetime.now()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(
                    str(now).replace(":", '')), fourcc, 20.0, (640, 480))
                # Start new thread for recording the video
                thread = Thread(target=record, args=[out, ])
                thread.start()
            elif(rec == False):
                out.release()

    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
