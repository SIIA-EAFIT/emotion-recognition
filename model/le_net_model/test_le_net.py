from le_net import LeNetModel
import os
import cv2
import numpy as np
import tensorflow as tf

tf.reset_default_graph()

model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),'..')
le_net_model_path = os.path.join(model_path, 'cnn_model')
model_dir = os.path.join(le_net_model_path, 'model')
project_path = os.path.join(model_path, '..')
data_path = os.path.join(project_path, 'data')
test_path = os.path.join(data_path, 'test')
test_image_dir = os.path.join(test_path, '4')
test_image = os.path.join(test_image_dir, '30') + '.png'


classifier = LeNetModel()
classifier.load_model(model_dir=model_dir)

im = cv2.imread(test_image,0)
im = im/255.0

predict = classifier.predict(im)
for result in predict:
    print(result)