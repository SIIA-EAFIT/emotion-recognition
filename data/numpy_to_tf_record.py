import tensorflow as tf
import sys
import h5py

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def from_hdf_to_tf_record(hdf_path, tf_record_path_and_name,
                          x_dict_name , y_dict_name, batch_size = 1000, ):
    
    dataset = h5py.File(hdf_path, 'r')
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(tf_record_path_and_name)
    n_examples , width , height = dataset[x_dict_name].shape
    
    for i in range(0 , n_examples, batch_size):

        # print how many images are saved every 1000 images
        if not i % 1000:
            print('Train data: {}/{}'.format(i, n_examples))
            sys.stdout.flush()
            
        # Load a batch from hdf and normalize it
        batch_data = dataset[x_dict_name][i:i+batch_size , : , : ] 
        batch_labels = dataset[y_dict_name][i:i+batch_size]

        for idx in range(len(batch_data)):
            img = batch_data[idx , : , : ] 
            label = batch_labels[idx]
            # Create a feature
            feature = {'label': _int64_feature(label),
                    'image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            
            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()