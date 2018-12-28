import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def to_numpy(data, normalize=True):
    data2 = data.split()
    if normalize:
            return np.array(data2, dtype='float32') / 255.0
    return np.array(data2, dtype='int32')

def save_tfrecord(data, writer, mirror=False):
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
        if mirror:
                x_mirror = np.reshape(x_numpy, (48, 48))
                x_mirror = np.flip(x_mirror, 1)
                x_mirror = x_mirror.flatten()
                x_mirror_bytes = x_mirror.tobytes()
                mirror_example = tf.train.Example(features=tf.train.Features(feature={
                        'label': _bytes_feature(y_bytes),
                        'image': _bytes_feature(x_mirror_bytes)
                }))
                writer.write(mirror_example.SerializeToString())

def save_image(data, data_dir, counter):
        for _, row in data.iterrows():
                image = to_numpy(row[1], normalize=False)
                image = np.reshape(image, (48,48))
                label = int(row[0])
                image_dir = os.path.join(data_dir, str(label))
                if not os.path.exists(image_dir):
                        os.makedirs(image_dir)
                image_name = os.path.join(image_dir, str(counter[label]))
                cv2.imwrite(image_name + '.png', image)
                counter[label] += 1
        return counter

data_dir = os.path.abspath(os.path.dirname(__file__))
train_dir = os.path.join(data_dir, 'train')

if not os.path.exists(train_dir):
        os.makedirs(train_dir)

validation_dir = os.path.join(data_dir, 'validation')

if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)

test_dir = os.path.join(data_dir, 'test')

if not os.path.exists(test_dir):
        os.makedirs(test_dir)

train_file = os.path.join(train_dir, 'train.tfrecord')
validation_file = os.path.join(validation_dir, 'validation.tfrecord')
csv_file = os.path.join(data_dir, 'fer2013.csv')

training_writer = tf.python_io.TFRecordWriter(train_file)
validation_writer = tf.python_io.TFRecordWriter(validation_file)

batch = 1000
cont = 0
counter = np.zeros(7).astype(np.int32)

for data in pd.read_csv(csv_file, header = None, chunksize=batch):
    print('evaluated ' + str(batch*cont) + ' images from dataset')
    training = data.loc[data[2] == 'Training']
    private_test = data.loc[data[2] == 'PrivateTest']
    public_test = data.loc[data[2] == 'PublicTest']
    save_tfrecord(training, training_writer, mirror=True)
    save_tfrecord(private_test, validation_writer)
    counter = save_image(public_test, test_dir, counter)
    cont +=1

training_writer.close()
validation_writer.close()
