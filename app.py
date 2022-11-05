from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import io
from io import StringIO
import base64
from PIL import Image
import cv2
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
from matplotlib import pyplot as plt

app = Flask(__name__)
socketio = SocketIO(app)

upperlip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 306, 292, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78, 62, 76, 61]
lowerlip = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 306, 292, 308, 324, 318, 402, 317, 14,87, 178, 88, 95, 78, 62, 76, 61]
alpha = 0.25
colorText = "color index: 0 - (w)"
colors = [cv2.COLORMAP_CIVIDIS, cv2.COLORMAP_DEEPGREEN, cv2.COLORMAP_INFERNO, cv2.COLORMAP_BONE, cv2.COLORMAP_MAGMA, cv2.COLORMAP_PLASMA]


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@socketio.on('image')
def image(data_image):
    sbuf = StringIO()
    sbuf.write(data_image[0])
    makeupIndex = data_image[1];
    b = io.BytesIO(base64.b64decode(data_image[0]))
    pimg = Image.open(b)
    frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)


    image = frame.copy()

    def createBox(img,points,scale=5,masked=False,cropped = True):
        if masked:
            mask = np.zeros_like(img)
            mask = cv2.fillPoly(mask,[points],(255,255,255))
            img = cv2.bitwise_and(img,mask)

            # cv2.imshow('Mask',img)

        if cropped:
            bbox = cv2.boundingRect(points)
            x,y,w,h = bbox
            imgCrop = img[y:y+h,x:x+w]
            imgCrop = cv2.resize(imgCrop,(0,0),None,scale,scale)
            return imgCrop
        else:
            return mask
    with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
        image.flags.writeable = False
        imgOriginal = image.copy()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (0,0), None, 0.5, 0.5)

        results = face_mesh.process(image)
        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        upperlipPoints = []
        lowerlipPoints = []

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0].landmark
            for i in upperlip:
                cord = _normalized_to_pixel_coordinates(face[i].x,face[i].y,imgOriginal.shape[1],imgOriginal.shape[0])
                upperlipPoints.append(cord)
                
            for i in lowerlip:
                cord = _normalized_to_pixel_coordinates(face[i].x,face[i].y,imgOriginal.shape[1],imgOriginal.shape[0])
                #cv2.circle(imgOriginal, (cord), 5, (50,50,255), cv2.FILLED)
                lowerlipPoints.append(cord)

        upperlipPoints = np.array(upperlipPoints)
        lowerlipPoints = np.array(lowerlipPoints)
        
        upperLipMask = createBox(imgOriginal, upperlipPoints, 3, masked=True, cropped=False)
        lowerLipMask = createBox(imgOriginal, lowerlipPoints, 3, masked=True, cropped=False)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40))

        upperLipMask = cv2.morphologyEx(upperLipMask, cv2.MORPH_CLOSE, kernel, 1)
        upperLipMask = cv2.GaussianBlur(upperLipMask,(15,15),cv2.BORDER_DEFAULT)

        inverseMaskUpper = cv2.bitwise_not(upperLipMask)
        upperLipMask = upperLipMask.astype(float)/255
        inverseMaskUpper = inverseMaskUpper.astype(float)/255#

        
        lowerLipMask = cv2.morphologyEx(lowerLipMask, cv2.MORPH_CLOSE, kernel, 1)

        lowerLipMask = cv2.GaussianBlur(lowerLipMask,(15,15),cv2.BORDER_DEFAULT)
        inverseMaskLower = cv2.bitwise_not(lowerLipMask)
        lowerLipMask = lowerLipMask.astype(float)/255
        inverseMaskLower = inverseMaskLower.astype(float)/255
        
        lips = cv2.applyColorMap(imgOriginal, colors[makeupIndex])
        

        lips = lips.astype(float)/255
        face = imgOriginal.astype(float)/255

        justLipsUpper = cv2.multiply(upperLipMask, lips)
        justLipsLower = cv2.multiply(lowerLipMask, lips)
        inverseMask = cv2.multiply(inverseMaskUpper, inverseMaskLower)


        justFace = cv2.multiply(inverseMask, face)


        result = justFace + justLipsUpper + justLipsLower

        font = cv2.FONT_HERSHEY_COMPLEX

        if cv2.waitKey(5) & 0xFF == 27:
            print("waiting")
        else:
            imgencode = cv2.imencode('.jpg', result*255)[1]

            stringData = base64.b64encode(imgencode).decode('utf-8')
            b64_src = 'data:image/jpg;base64,'
            stringData = b64_src + stringData

            emit('response_back', stringData)
    

socketio.run(app, host='127.0.0.1', port=5000)