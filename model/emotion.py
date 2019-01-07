from tensorflow.estimator import Estimator
import tensorflow as tf
import numpy as np
import h5py
import sys

import alex_net

sys.path.append("..") # Adds higher directory to python modules path.

from data.__init__ import Dataset
from data import utils_functs
from data  import numpy_to_tf_record as ntr



class Emotion:

    def __init__(self, download = False):

        self.hdf_dataset = '../data/dataset/dataset.hdf5'
        self.tf_training_record_path = '../data/training_record'
        self.tf_validation_record_path = '../data/validation_record'
        self.model_dir = '../data/tensorflow-graph'
        #If the dataset needs to be downloaded
        if (download):
            self.dataset = Dataset(download=True, gen_images=True, gen_hdf5=True)

        self.load_data()



    def load_data(self):

        """Loading data from the hdf5 file
        
        """
        ntr.from_hdf_to_tf_record(self.hdf_dataset, self.tf_training_record_path
        , x_dict_name = "train_img" , y_dict_name = "train_labels"
        )

        ntr.from_hdf_to_tf_record(self.hdf_dataset ,self.tf_validation_record_path , 
            x_dict_name = "val_img" , y_dict_name = "val_labels"
        )


        self.main()

    def main(self, 
            steps=None, batch_size = 300 , epochs = 1 , learning_rate = 0.01 , optimizer = None, 
            number_of_classes = 7, imgWidth = 48, imgHeight = 48 , n_channels = 1) :

        if optimizer is None:
             optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        config = tf.estimator.RunConfig(
                save_summary_steps=10
            
        )
            

        classifier = Estimator(
            model_fn= alex_net.model,
            params={
                'optimizer' : optimizer,
                'number_of_classes' : number_of_classes,
                'batch_size' : batch_size,
                'imgWidth' : imgWidth,
                'imgHeight' : imgHeight,
                'n_channels' : n_channels
                
            },
            config=config,
            model_dir=self.model_dir
            )

        classifier.train(
        input_fn=lambda: utils_functs._input_fn(filename = self.tf_training_record_path, 
                                            batch_size = batch_size,
                                            epochs = epochs),
        steps=steps)
        
        eval_result = classifier.evaluate(
        input_fn=lambda: utils_functs._input_fn(filename = self.tf_validation_record_path, 
                                            batch_size = batch_size,
                                            epochs = epochs))

        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


    
    

emotion = Emotion()