from le_net import LeNetModel
import tensorflow as tf
import os
tf.reset_default_graph()

model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
le_net_model_path = os.path.join(model_path, 'le_net_model')
model_dir = os.path.join(le_net_model_path, 'model')
project_path = os.path.join(model_path, '..')
data_path = os.path.join(project_path, 'data')
train_path = os.path.join(data_path, 'train')
validation_path = os.path.join(data_path, 'validation')

model = LeNetModel()

train_file = os.path.join(train_path, 'train.tfrecord')
validation_file = os.path.join(validation_path, 'validation.tfrecord')

train_data = tf.data.TFRecordDataset(train_file)
validation_data = tf.data.TFRecordDataset(validation_file)

model.train(train_data, validation_data, model_dir=model_dir, learning_rate=0.001, epochs=30)