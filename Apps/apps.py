import cv2
import os, shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

from flask import Flask, render_template, request, redirect, url_for, json
from PIL import Image

PATH_MODEL_WEIGHT = './resources/model_weight/weight.zip'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")

if(device.type == 'cuda'):
    torch.cuda.empty_cache()

class Haardcascade:
    cascade = 'resources/haarcascade/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade)  
    
    @staticmethod
    def preprocess(image_bgr):
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        image = cv2.equalizeHist(image)
        return image

    @classmethod
    def detect(cls, image_bgr):
        gray_image = cls.preprocess(image_bgr)
        bounding_boxes = cls.face_cascade.detectMultiScale(gray_image, 1.05, 3)
        return bounding_boxes

PATH_SOURCE = './static/source.jpg'
PATH_PREDICT = './static/result.jpg'
PATH_CROP = './static/crop.jpg'

emotions = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

image_shape = (224, 224)
transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize(image_shape),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def init_model(pretrained=True):
    model = models.vgg16(pretrained=pretrained)
    model.classifier[6] = nn.Linear(4096, 7)
    return model

def load_weight(model, path):
    model.load_state_dict(torch.load(path))
    print(f"Loaded model weight from {path}")

model = init_model(pretrained=False)
load_weight(model, PATH_MODEL_WEIGHT)
model.to(device)

LABELS = json.dumps( list(emotions) )
DATA = None

def resize(image):
    width = int(512)
    height = int(image.shape[0] * (width / image.shape[1]))
    dim = (width, height)
    image = cv2.resize(image, dim)
    return image

def process():
    
    # Resize source image
    image = cv2.imread(PATH_SOURCE)
    image = resize(image)
    cv2.imwrite(PATH_SOURCE, image)

    # Get single bounding boxes
    bounding_boxes = Haardcascade.detect(image)
    bounding_box = (0,0,1,1)
    for (x,y,w,h) in bounding_boxes:
        area = (w*h)
        max_area = bounding_box[2] * bounding_box[3]
        if(area > max_area):
            bounding_box = (x,y,w,h)

    x, y, w, h = bounding_box

    # Save and Draw bounding box
    draw_image = image.copy()
    cv2.rectangle(draw_image, (x,y), (x+w, y+h), (0,255,0), 1)
    cv2.imwrite(PATH_PREDICT, draw_image)

    # Crop bounding box
    crop_image = image[y:y+h, x:x+w]
    cv2.imwrite(PATH_CROP, crop_image)

    del x, y, w, h

    # Predict logits
    image = Image.open(PATH_CROP)
    image = transform(image)
    image = image.to(device)
    logits = model(image.view(1, image.shape[0], image.shape[1], image.shape[2]))[0]
    pred = F.softmax(logits)
    global DATA
    DATA = json.dumps( list(pred.data.cpu().numpy().astype('float')) )

# ======================================================================

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    if(request.method == 'GET'):
        return render_template('index.html')
    elif(request.method == 'POST'):
        image = request.files['image']
        image.save(PATH_SOURCE)
        process()
        return redirect(url_for('display'))

@app.route('/result', methods=['GET'])
def display():
    # print(f"Labels : {LABELS}")
    # print(f"Logits: {DATA}")
    return render_template('display.html', data=DATA, labels=LABELS)

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
    shutil.rmtree('static')