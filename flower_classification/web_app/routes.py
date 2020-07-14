from web_app import app

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow_hub as hub

import os
from keras.preprocessing import image
import uuid
import base64
from flask import send_from_directory
import PIL
from PIL import Image
import numpy as np


model = load_model('1594207945.h5', custom_objects={'KerasLayer':hub.KerasLayer})


upload_dir="web_app/static/uploaded"
app.config['UPLOAD_FOLDER'] = upload_dir

ALLOWED_EXTENSIONS=set(['png', 'jpg', 'jpeg', 'bmp'])


@app.route('/')
def upload_f():
    return render_template('pred.html', imagesource="/static/g_image.png", ss='')

def finds(path):
    
    vals = ['Pink primrose', 'Hard-leaved pocket orchid', 'Canterbury bells', 'Sweet pea',
            'English marigold', 'Tiger lily', 'Moon orchid', 'Bird of paradise', 'Monkshood',
            'Globe thistle', 'Snapdragon', "Colt's foot", 'King protea', 'Spear thistle',
            'Yellow iris', 'Globe-flower', 'Purple coneflower', 'Peruvian lily', 
            'Balloon flower', 'Giant white arum lily', 'Fire lily, Pincushion flower', 
            'Fritillary', 'Red ginger','Grape hyacinth', 'Corn poppy', 
            'Prince of wales feathers', 'Stemless gentian','Artichoke', 'Sweet william', 
            'Carnation', 'Garden phlox', 'Love in the mist','Mexican aster', 
            'Alpine sea holly', 'Ruby-lipped cattleya', 'Cape flower','Great masterwort', 
            'Siam tulip', 'Lenten rose', 'Barbeton daisy', 'Daffodil','Sword lily', 
            'Poinsettia', 'Bolero deep blue', 'Wallflower', 'Marigold', 'Buttercup', 
            'Oxeye daisy', 'Common dandelion', 'Petunia', 'Wild pansy', 'Primula', 
            'Sunflower', 'Pelargonium', 'Bishop of llandaff', 'Gaura','Geranium',
            'Orange dahlia', 'Pink-yellow dahlia', 'Cautleya spicata','Japanese anemone',
            'Black-eyed susan', 'Silverbush', 'Californian poppy','Osteospermum',
            'Spring crocus', 'Bearded iris', 'Windflower', 'Tree poppy','Gazania, Azalea',
            'Water lily', 'Rose', 'Thorn apple', 'Morning glory','Passion flower',
            'Lotus lotus', 'Toad lily', 'Anthurium', 'Frangipani','Clematis', 'Hibiscus',
            'Columbine', 'Desert-rose', 'Tree mallow','Magnolia', 'Cyclamen', 'Watercress',
            'Canna lily', 'Hippeastrum', 'Bee balm', 'Ball moss', 'Foxglove',
            'Bougainvillea', 'Camellia','Mallow', 'Mexican petunia', 'Bromelia',
            'Blanket flower', 'Trumpet creeper', 'Blackberry lily']
    
    image_shape=299
    img=image.load_img(path, target_size=(image_shape, image_shape))
    img_arr=image.img_to_array(img)
    img_to_pred=np.expand_dims(img_arr, axis=0)
        
    prediction = model.predict(img_to_pred)

    prediction=tf.squeeze(prediction).numpy()
    predict_id=np.argmax(prediction)
    print("PREDICTION: ",predict_id)
    print(vals[int(predict_id)])
    return vals[int(predict_id)]


def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() 
    random = random.replace("-","") 
    return random[0:string_length] 


def allowed_file(filename):
    return "." in filename and \
    filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/uploaded', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            val = finds(file_path)

            filename = my_random_string(6) + filename
            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
         
            return render_template('pred.html', ss = val, imagesource="static/uploaded/"+filename)


@app.route('/uploaded/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)




