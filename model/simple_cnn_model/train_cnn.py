from cnn_model import CNNModel
import tensorflow as tf
import os
tf.reset_default_graph()

model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),'..')
cnn_model_path = os.path.join(model_path, 'simple_cnn_model')
model_dir = os.path.join(cnn_model_path, 'model')
project_path = os.path.join(model_path, '..')
data_path = os.path.join(project_path, 'data')
train_path = os.path.join(data_path, 'train')
validation_path = os.path.join(data_path, 'validation')

model = CNNModel()

train_file = os.path.join(train_path, 'train.tfrecord')
validation_file = os.path.join(validation_path, 'validation.tfrecord')

train_data = tf.data.TFRecordDataset(train_file)
validation_data = tf.data.TFRecordDataset(validation_file)

print('Model dir: ', model_dir)

model.train(train_data, validation_data, model_dir=model_dir, learning_rate=0.0001, epochs=30)