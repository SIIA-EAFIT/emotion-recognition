from tensorflow.estimator import DNNClassifier
import tensorflow as tf
import numpy as np


class DNNModel:

    def __init__(self):
        self.classifier = None

    def train(self, train_data, validation_data=None,
              hidden_units=[1024, 512, 256, 128],
              learning_rate=0.01, batch_size=500, model_dir=None,
              epochs=10, n_classes=7, checkpoint_path=None,
              save_checkpoint_steps=5, keep_checkpoint_max=1):
        training_function = self.input_fn(train_data, batch_size, shuffle=True)
        evaluate_training_fn = self.input_fn(train_data, batch_size)
        evaluate_validation_fn = None
        if validation_data:
            evaluate_validation_fn = self.input_fn(validation_data, batch_size)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        config = tf.estimator.RunConfig(
            save_checkpoints_steps=save_checkpoint_steps,
            keep_checkpoint_max=keep_checkpoint_max)
        
        self.classifier = DNNClassifier(
            feature_columns=self._feature_columns(),
            hidden_units=hidden_units,
            n_classes=n_classes,
            optimizer=optimizer,
            config=config,
            model_dir=model_dir,
            warm_start_from=checkpoint_path)

        print("Training model...")
        for epoch in range(epochs):
            self.classifier.train(input_fn=training_function)
            print("Epoch: ", epoch)
            
            training_metrics = self.classifier.evaluate(input_fn=evaluate_training_fn, name='training')
            print("training", training_metrics)
            if validation_data:
                validation_metrics = self.classifier.evaluate(input_fn=evaluate_validation_fn, name='validation')
                print("validation", validation_metrics)

    def load_model(self, model_dir, hidden_units=[1024, 512, 256, 128], n_classes=7):
        self.classifier = DNNClassifier(feature_columns=self._feature_columns(),
                                           hidden_units=hidden_units,
                                           model_dir=model_dir, n_classes=n_classes)
        self.classifier.latest_checkpoint()

    def predict(self, X, tfrecord=False):
        predict_fn = None
        if tfrecord:
            predict_fn = self.input_fn(X,batch_size=250)
        else:
            predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"image": X},
            num_epochs=1,
            shuffle=False)
            predict_fn = predict_input_fn
        return self.classifier.predict(input_fn=predict_fn)

    def input_fn(self, data, batch_size, shuffle=False):
        def _input_fn():
            train_data = data.map(self._extract_fn)
            if shuffle:
                train_data = train_data.shuffle(10000)
            train_data = train_data.batch(batch_size)
            train_data = train_data.prefetch(1)
            iterator = train_data.make_one_shot_iterator()
            image, label = iterator.get_next()
            return image, label

        return _input_fn

    def _predict_fn(self, X):
        def _input_fn():
            features = {}
            features['image'] = X
            dataset = tf.data.map(features)
            iterator = dataset.make_one_shot_iterator()
            image = iterator.get_next()
            return image

        return _input_fn

    def _feature_columns(self):
        return {tf.feature_column.numeric_column('image', shape=(1,2304))}
          
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