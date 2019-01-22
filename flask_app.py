from flask import Flask, render_template, request, jsonify
import cv2
import base64
import os
import io
import sys
import numpy as np
from imageio import imread

from model.emotion import Emotion

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/", methods=['GET'])
def inicio():
    return render_template('Main.html')


@app.route("/about")
def montaje():
    return render_template('about.html')


@app.route("/tryIt")
def tryIt():
    return render_template('load_image.html')


@app.route("/upload" , methods=['POST'])
def upload():

    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if(not os.path.isdir(target)):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target,filename])
        print(destination)
        file.save(destination)

    return render_template('load_image.html')

@app.route("/saveImage" , methods=['POST'])
def image():
    result_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

    if(request.get_json()):
        image = request.get_json()[22:]  #removing the image header
        decoded_image = base64.b64decode(image)
        img = imread(io.BytesIO(decoded_image))

     
        img = cv2.resize(img , (48,48))
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = img.astype(np.float32)

        emotion = Emotion()
        emotion.load_model('data/tensorflow-graph')
        results = emotion.predict( img , False)
        result = next(results)
        _class = result_dict[result["classes"]]
        probabilities = result["probabilities"]

        #This code is useful if the image is needed to be saved as jpg instead of a vector
        # image_to_be_saved = "./images/current_image.jpg"
        # with open(image_to_be_saved,'wb' ) as img:
        #      img.write(decoded_image)

        resp = jsonify(success=True , result = _class, prob = probabilities.tolist())
    else:
        resp = jsonify(success=False)

    
    return resp

if __name__ == '__main__':
    #app.run(host='0.0.0.0', debug=True, port=8000)
    app.run(debug=True, port=8000)
