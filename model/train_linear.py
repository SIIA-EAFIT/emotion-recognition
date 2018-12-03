from linear_model import Linear_Model
import tensorflow as tf
import os

model_path = os.path.abspath(os.path.dirname(__file__))
linear_model_path = os.path.join(model_path, 'linear_model')
save_model_dir = os.path.join(linear_model_path, 'model')
project_path = os.path.join(model_path, '..')
data_path = os.path.join(project_path, 'data')
train_path = os.path.join(data_path, 'train')
validation_path = os.path.join(data_path, 'validation')

model = Linear_Model(n_classes=7, save_dir=save_model_dir)

train_file = os.path.join(train_path, 'train.tfrecord')
validation_file = os.path.join(validation_path, 'validation.tfrecord')

train_data = tf.data.TFRecordDataset(train_file)
validation_data = tf.data.TFRecordDataset(validation_file)

classifier = model.train(train_data, validation_data)