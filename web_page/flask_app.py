from flask import Flask, render_template, request
import os
import sqlite3 as sql

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


if __name__ == '__main__':
    #app.run(host='0.0.0.0', debug=True, port=8000)
    app.run(debug=True, port=8000)
