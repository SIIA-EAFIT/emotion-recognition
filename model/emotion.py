from tensorflow.estimator import Estimator
import tensorflow as tf
import numpy as np
import h5py
import sys
import cv2

import os

import model.alex_net as alex_net

sys.path.append("..")  # Adds previous directory to python modules path.

from data.__init__ import Dataset
from data import utils_functs
from data import numpy_to_tf_record as ntr


class Emotion:
    def __init__(self, download=False, load_data=False):
        self.classifier = None
        self.hdf_dataset = '../data/dataset/dataset.hdf5'
        self.tf_training_record_path = '../data/training_record'
        self.tf_validation_record_path = '../data/validation_record'
        self.model_dir = '../data/tensorflow-graph'
        #If the dataset needs to be downloaded
        if (download):
            self.dataset = Dataset(
                download=True, gen_images=True, gen_hdf5=True)
        if (load_data):
            self.load_data()

    def load_data(self):
        """Loading data from the hdf5 file        
        """
        ntr.from_hdf_to_tf_record(
            self.hdf_dataset,
            self.tf_training_record_path,
            x_dict_name="train_img",
            y_dict_name="train_labels")

        ntr.from_hdf_to_tf_record(
            self.hdf_dataset,
            self.tf_validation_record_path,
            x_dict_name="val_img",
            y_dict_name="val_labels")

    def predict(self, examples, tfrecord=False):
        predict_fn = None
        if tfrecord:
            predict_fn = lambda: utils_functs._input_fn(
                examples, batch_size=250, epochs=1)
        else:
            predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x=examples, shuffle=False)
            predict_fn = predict_input_fn

        return self.classifier.predict(input_fn=predict_fn)

    def load_model(self, model_dir):
        self.classifier = Estimator(
            model_fn=alex_net.model,
            params={
                'number_of_classes': 7,
                'batch_size': 1,
                'imgWidth': 48,
                'imgHeight': 48,
                'n_channels': 1
            },
            model_dir=model_dir)
        self.classifier.latest_checkpoint()

    def train(self,
              batch_size=500,
              epochs=10,
              steps=None,
              learning_rate=0.01,
              optimizer=None,
              number_of_classes=7,
              imgWidth=48,
              imgHeight=48,
              n_channels=1):

        if optimizer is None:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        config = tf.estimator.RunConfig(save_summary_steps=10)

        self.classifier = Estimator(
            model_fn=alex_net.model,
            params={
                'optimizer': optimizer,
                'number_of_classes': number_of_classes,
                'batch_size': batch_size,
                'imgWidth': imgWidth,
                'imgHeight': imgHeight,
                'n_channels': n_channels
            },
            config=config,
            model_dir=self.model_dir)


        self.classifier.train(
        input_fn=lambda: utils_functs._input_fn(filename = self.tf_training_record_path,
                                            batch_size = batch_size,
                                            epochs = epochs),
        steps=steps)

        self.eval_result = self.classifier.evaluate(
        input_fn=lambda: utils_functs._input_fn(filename = self.tf_validation_record_path,
                                            batch_size = batch_size,
                                            epochs = epochs, evaluation = True))

        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(
            **self.eval_result))
    

# result_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}


# emotion = Emotion()
# #emotion.train()
# emotion.load_model('../data/tensorflow-graph')
# test_image = '../data/dataset/0/1234.jpg'
# im = cv2.imread(test_image,0)
# im = im.astype(np.float32)
# results = emotion.predict( im , False)
# print(next(results)['classes'])
# for result in results:
#     print(result_dict[result['classes']])