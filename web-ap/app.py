import os
import datetime
import random

from flask import Flask, render_template, request, redirect
from flask.helpers import url_for

from config import config

import cv2
from collections import defaultdict
from recognitor import initialize_recognitor
import numpy as np
import imutils

import time

app = Flask(__name__)


predictor, model = initialize_recognitor()

mapping = {'не дефект': 0,
             'потертость': 1,
             'черная точка': 2,
             'плена': 3,
             'маркер': 4,
             'грязь': 5,
             'накол': 6,
             'н.д. накол': 7,
             'микровыступ': 8,
             'н.д. микровыступ': 9,
             'вмятина': 10,
             'мех.повреждение': 11,
             'риска': 12,
             'царапина с волчком': 13}

mapping = {v: k for k, v in mapping.items()}

def allowed_file(filename):
    if filename.rsplit('.', 1)[1].lower() in config['ALLOWED_EXTENSIONS']:
        return True
    else:
        return False


@app.route('/', methods=['POST', 'GET'])
def index():

    if request.method == 'GET':
        global curr_index
        global form_text
        global LIST_IMAGES
        curr_index = 0
        form_text = ''
        return render_template('index.html')


@app.route('/get_folder', methods=['POST', 'GET'])
def get_folder():
    
    if request.method == 'GET':
        global curr_index
        curr_index += 1
    
    if 'text' in request.form.keys():
        global form_text
        form_text = str(request.form['text'])
        global LIST_IMAGES
        LIST_IMAGES =  os.listdir(os.path.join('/home/services/flask-app-template/static/', form_text))
        
    try:
        img_path = os.path.join('static', form_text, LIST_IMAGES[curr_index])
    except:
        #TO DO Конец обработки всей папки
        img_path = os.path.join('static', form_text, LIST_IMAGES[0])
    print(img_path)
    pred_img, out = predict(img_path)
    cv2.imwrite(f"./static/img/results/1.jpg", pred_img)
    time.sleep(0.2)
    columns_gt = ['Найденный дефект', 'X', 'Y']
    items_gt = out
    
    return render_template('recognized.html', img_path='/img/results/1.jpg', items_gt=items_gt, columns_gt=columns_gt)


@app.route('/get_image', methods=['POST', 'GET'])
def get_image():
          
    if 'file' not in request.files:
        error = True
        return render_template('index.html', error=error)

    file = request.files['file']
    
    if file.filename == '':
        error = True
        return render_template('index.html', error=error)
    

    if file and allowed_file(file.filename):
        
        image = request.files['file']
        image_filename = config['IMAGE_NAME']
        path = os.path.join(config['UPLOADS_PATH'], image_filename)

        try:
            if not os.path.exists(config['UPLOADS_PATH']):
                os.makedirs(config['UPLOADS_PATH'])
            image.save(path)
        except:
            error = True
            return render_template('index.html', error=error)
    else:
        error = True
        return render_template('index.html', error=error)
  
    pred_img, out = predict(path)
    cv2.imwrite(f"./static/img/results/1.jpg", pred_img)
    time.sleep(0.2)
    columns_gt = ['Найденный дефект', 'X', 'Y']
    items_gt = out
    
    return render_template('recognized_img.html', img_path='/img/results/1.jpg', items_gt=items_gt, columns_gt=columns_gt)

  
def draw_mask(image, mask, alpha=0.3):
    img = image.copy()
    img = cv2.addWeighted(img, 1, mask, alpha, 0)
    return img

def predict(image_path):
    visualize = False
    out = []
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predict_img = image.copy()
    
    results = model.predict(image_path, device='cuda:1')
    result = results[0]
    
    img_h, img_w = result.orig_shape
    masks = np.zeros((img_h, img_w), dtype=np.uint8)
    
    predicts = defaultdict(list)
    for box in result.boxes:
        x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
        x = int(x1 + (x2 - x1) / 2)
        y = int(y1 + (y2 - y1) / 2)
        
        conf = round(box.conf[0].item(), 2)
        if conf < 0.25:
            continue
        
        class_id = int(box.cls[0].item()) + 1
        predicts[class_id].append([x, y])
        

        out.append({'Найденный дефект': mapping[class_id], 'X': str(x), 'Y': str(y)})

        predict_img = cv2.circle(predict_img, (x, y), radius=5, color=(255, 0, 255), thickness=-1)
     
        
    for label, points in predicts.items():
        predictor.set_image(image)
        mask, _, _ = predictor.predict(
            point_coords=np.array(points),
            point_labels=[label] * len(points),
            box=None, 
            multimask_output=True,
        )
        masks[mask[0]] = class_id #* 18
    
    u_labels = np.unique(masks)
    for u_label in u_labels:
        if u_label == 0:
            continue
        mask = (masks == u_label).astype(np.uint8) * 255
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        for c in cnts:
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
      
            predict_img = cv2.drawContours(predict_img, [c], -1, (0, 128, 255), 2)
            predict_img = cv2.circle(predict_img, (cX, cY), 7, (32, 32, 255), -1)
    
    mask = np.stack([np.zeros_like(masks), np.zeros_like(masks), masks]).transpose(1, 2, 0)
    predict_img = draw_mask(predict_img, mask, 0.7)


    for box in result.boxes:
        x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
        x = int(x1 + (x2 - x1) / 2)
        y = int(y1 + (y2 - y1) / 2)
        
        conf = round(box.conf[0].item(), 2)
        if conf < 0.25:
            continue
        
        class_id = int(box.cls[0].item()) + 1
        predict_img = cv2.putText(predict_img, str(class_id), (x - 5, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.9, (128, 0, 0), 2)

    return predict_img, out

if __name__ == '__main__':
    app.run(debug=False, port=8705)
