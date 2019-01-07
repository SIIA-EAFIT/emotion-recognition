import tensorflow as tf 


def _input_fn(filename , batch_size, epochs):
        
    """An input function for training

    Taken from : https://www.tensorflow.org/guide/custom_estimators
    
    """

    # Convert the inputs to a Dataset
    dataset = tf.data.TFRecordDataset(filename)

    dataset = dataset.map(parse)

    # repeat, and batch the examples. Training data is already shuffled
    dataset = dataset.repeat(epochs).batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()




def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""

    features = tf.convert_to_tensor(features , dtype = tf.float64)
    labels = tf.convert_to_tensor(labels ,dtype = tf.int32)
    
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

def parse(serialized):
    """
    This function parses a single serialized example of a tfrecord file. 
    ----------------------------------Taken from Sebastian Colinas

    Args:
        serialized: serialized example.

    Returns:
        A tuple of image, label

    """
    features = \
        {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized,
                                                features=features)
    # Get the image as raw bytes.
    image_raw = parsed_example['image']

    # Decode the raw bytes so it becomes a tensor with type.
    image = tf.decode_raw(image_raw, tf.uint8)

    # The type is now uint8 but we need it to be float.
    image = tf.cast(image, tf.float32)

    # Get the label associated with the image.
    label = parsed_example['label']

    # The image and label are now correct TensorFlow types.
    return image, label
