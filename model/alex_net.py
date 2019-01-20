import tensorflow as tf


def model(features , labels , mode, params):

        img_height = params['imgHeight']
        img_width = params['imgWidth']
        n_channels = params['n_channels']
        n_classes = params['number_of_classes']
        batch_size = params['batch_size']
        training = mode == tf.estimator.ModeKeys.TRAIN
     
        ################################################################################
        

        #Input and dealing with the first dimension which is the  unknown batch size
        net = tf.keras.Input( shape = (  img_height , img_width ) , tensor = features, batch_size = batch_size)
        net = tf.reshape(features, shape=( -1 , img_height, img_width, n_channels))
        
        #Conv -> Activation -> Maxpooling
        net = tf.layers.conv2d(inputs= net , filters=16, strides=(2,2) , kernel_size=(5, 5), 
                                 activation=tf.nn.relu, name="conv_0")
       
        #Conv -> Activation
        net = tf.layers.conv2d(inputs= net , filters=48, strides=(1,1) , kernel_size=(3, 3), 
                                 activation=tf.nn.relu, name="conv_1")  
        
        #Conv -> Activation  ; Conv -> Activation
        net = tf.layers.conv2d(inputs= net , filters=96, strides=(1,1) , kernel_size=(3, 3), 
                                 activation=tf.nn.relu, name="conv_2")
        net = tf.layers.conv2d(inputs= net , filters=96, strides=(1,1) , kernel_size=(3, 3), 
                                 activation=tf.nn.relu, name="conv_3")
        
        #Conv -> Activation -> Maxpooling
        net = tf.layers.conv2d(inputs= net , filters=48, strides=(1,1) , kernel_size=(3, 3), 
                                 activation=tf.nn.relu, name="conv_4")  
        net = tf.layers.max_pooling2d(inputs= net, pool_size=(2, 2), strides=(2, 2))
        
        #Flatten -> dropout
        net = tf.layers.flatten(inputs=net, name = "flatten")
        net = tf.layers.dropout(inputs=net, rate=0.2, training = training)
        
        #Dense -> ReLU -> Dropout -> Dense -> Relu -> Logits
        net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.relu, name='fc1')    
        net = tf.layers.dropout(inputs=net, rate=0.2, training = training)
        net = tf.layers.dense(inputs=net, units=1024,  activation=tf.nn.relu, name='fc2')   

        #logits
        net = tf.layers.dense(inputs=net, units=n_classes, name='fc3')
        ################################################################################
        
        predictions = {
            "classes": tf.argmax(input=net, axis=1),
            "probabilities": tf.nn.softmax(net, name="softmax")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=net)
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = params["optimizer"]
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            
            with tf.variable_scope('training_performance'):
                tf.summary.scalar('loss', loss)
                tf.summary.scalar('accuracy', tf.metrics.accuracy(labels, predictions["classes"])[1])

        
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
        }

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)            



