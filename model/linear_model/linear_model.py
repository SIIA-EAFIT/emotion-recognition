from tensorflow.estimator import LinearClassifier
import tensorflow as tf
import numpy as np

class Linear_Model:

    def __init__(self, learning_rate=0.01, batch_size=500,
                 periods=10, n_classes=7,
                 save_dir=None,
                 checkpoint_path=None,
                 save_checkpoint_steps=5,
                 keep_checkpoint_max=1):
        self.learning_rate = learning_rate
        self.model = None
        self.estimator = None
        self.batch_size = batch_size
        self.periods = periods
        self.n_classes = n_classes
        self.save_dir = save_dir
        self.checkpoint_path = checkpoint_path
        self.save_checkpoint_steps = save_checkpoint_steps
        self.keep_checkpoint_max = keep_checkpoint_max

    def train(self, train_data, validation_data=None):
        training_function = self.input_fn(train_data, shuffle=True)
        evaluate_training_fn = self.input_fn(train_data)
        evaluate_validation_fn = None
        if validation_data:
            evaluate_validation_fn = self.input_fn(validation_data)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        config = tf.estimator.RunConfig(
            save_checkpoints_steps=self.save_checkpoint_steps,
            keep_checkpoint_max=self.keep_checkpoint_max)
        
        classifier = LinearClassifier(
            feature_columns=self._feature_columns(),
            n_classes=self.n_classes,
            optimizer=optimizer,
            config=config,
            model_dir=self.save_dir,
            warm_start_from=self.checkpoint_path)

        print("Training model...")
        for period in range(self.periods):
            classifier.train(input_fn=training_function)
            print("Period: ", period)
            
            training_metrics = classifier.evaluate(input_fn=evaluate_training_fn, name='training')
            print("training", training_metrics)
            if validation_data:
                validation_metrics = classifier.evaluate(input_fn=evaluate_validation_fn, name='validation')
                print("validation", validation_metrics)

    def predict(self, X, tfrecord=False):
        predict_fn = None
        if(tfrecord):
            predict_fn = self.input_fn(X)
        else:
            predict_fn = self._predict_fn(X)
        self.estimator.predict(input_fn=predict_fn)

    def input_fn(self, data, shuffle=False):
        def _input_fn():
            train_data = data.map(self._extract_fn)
            if shuffle:
                train_data = train_data.shuffle(10000)
            train_data = train_data.batch(self.batch_size)
            train_data = train_data.prefetch(1)
            iterator = train_data.make_one_shot_iterator()
            image, label = iterator.get_next()
            return image, label

        return _input_fn

    def _predict_fn(self, X):
        def _input_fn():
            dataset = tf.data.Dataset.from_tensor_slices(X)
            iterator = dataset.make_one_shot_iterator()
            image = iterator.get_next()
            return image

        return _input_fn

    def _feature_columns(self):
        return set([tf.feature_column.numeric_column('image', shape=2304)])
          
    def _extract_fn(self, data_record):
        features_keys = {
            'label': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string),
        }
        features = {}
        sample = tf.parse_single_example(data_record, features_keys)
        image = tf.decode_raw(sample['image'], tf.float32)
        label = tf.decode_raw(sample['label'], tf.int32)
        features['image'] = image
        return features, label
