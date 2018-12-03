import pandas as pd
import numpy as np
import os
import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def to_numpy(data):
    data2 = data.split()
    return np.array(data2, dtype='float32') / 255.0

def save_data(data, writer):
    for index, row in data.iterrows():
        x_numpy = to_numpy(row[1])
        y = np.zeros(1, dtype=np.int32)
        y[0] = int(row[0])
        x_bytes = x_numpy.tobytes()
        y_bytes = y.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
                'label': _bytes_feature(y_bytes),
                'image': _bytes_feature(x_bytes)
                }))
        writer.write(example.SerializeToString())

data_dir = os.path.abspath(os.path.dirname(__file__))
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'validation')
test_dir = os.path.join(data_dir, 'test')

train_file = os.path.join(train_dir, 'train.tfrecord')
validation_file = os.path.join(validation_dir, 'validation.tfrecord')
test_file = os.path.join(test_dir, 'test.tfrecord')
csv_file = os.path.join(data_dir, 'fer2013.csv')

training_writer = tf.python_io.TFRecordWriter(train_file)
validation_writer = tf.python_io.TFRecordWriter(validation_file)
test_writer = tf.python_io.TFRecordWriter(test_file)

batch = 1000
cont = 0

for data in pd.read_csv(csv_file, header = None, chunksize=batch):
    print('evaluated ' + str(batch*cont) + ' images from dataset')
    training = data.loc[data[2] == 'Training']
    private_test = data.loc[data[2] == 'PrivateTest']
    public_test = data.loc[data[2] == 'PublicTest']
    save_data(training, training_writer)
    save_data(private_test, validation_writer)
    save_data(public_test, test_writer)
    cont +=1

training_writer.close()
validation_writer.close()
test_writer.close()