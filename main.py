import imghdr
import os
from fastai.vision.all import *
import torchvision.transforms as T
import sys
from flask import Flask, render_template, request, redirect, url_for, abort, \
    send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'static/uploads'

def validate_image(stream):
    header = stream.read(512)  # 512 bytes should be enough for a header check
    stream.seek(0)  # reset stream pointer
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

def deleteImages():
    for f in os.listdir(app.config['UPLOAD_PATH']):
        os.remove(os.path.join(app.config['UPLOAD_PATH'], f))

def predict(img_path):
    learn = load_learner('static/model1.pkl')
    pred_classes, pred_idx, probs = learn.predict(img_path)
    return pred_classes, pred_idx, probs

@app.route('/')
def index():
    if os.listdir(app.config['UPLOAD_PATH']):
        deleteImages()
    files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('index.html', files=files)

@app.route('/', methods=['POST'])
def upload_files():
    if os.listdir(app.config['UPLOAD_PATH']):
        deleteImages()
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                file_ext != validate_image(uploaded_file.stream):
            abort(400)
        img_path = os.path.join(app.config['UPLOAD_PATH'], filename)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
        pred_classes, pred_idx, probs = predict(img_path)
        pred_classes = ', '.join([x.capitalize() for x in pred_classes])
        probs = ', '.join([str(round(x, 4)) for x in probs[pred_idx].tolist()])
        files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('index.html', files=files, pred_class=pred_classes,
        outputs=str(probs))

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)