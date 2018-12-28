from dnn_model import DNNModel
import os
import cv2
import numpy as np
import tensorflow as tf

tf.reset_default_graph()

model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
dnn_model_path = os.path.join(model_path, 'dnn_model')
model_dir = os.path.join(dnn_model_path, 'model')
project_path = os.path.join(model_path, '..')
data_path = os.path.join(project_path, 'data')
test_path = os.path.join(data_path, 'test')
test_image_dir = os.path.join(test_path, '0')
test_image = os.path.join(test_image_dir, '15') + '.png'


classifier = DNNModel()
classifier.load_model(model_dir=model_dir)

im = cv2.imread(test_image,0)
im = im.flatten()
im = np.reshape(im, (1,2304))
im = im/255.0
print('image shape: ', im.shape)

predict = classifier.predict(im)
for result in predict:
    print(result)