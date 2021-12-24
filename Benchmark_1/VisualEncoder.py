import tensorflow as tf
import numpy as np
from abc import ABC, abstractclassmethod

class VisualEncoder(ABC):
    def __init__(self, data, dense_len):
        self.data_ = data
        self.dense_len_ = dense_len
        self.dense_ = None

    @property
    def dense(self):
        assert(self.dense_ is not None)
        return self.dense_

class FcEncoder(VisualEncoder):
    '''
    data: a tensor (placeholder)
    '''
    def __init__(self, data, dense_len, data_len):
        super().__init__(data, dense_len)
        self.data_len_ = data_len
        # with tf.variable_scope('FcEncoder'):
        # self.layer_ = tf.layers.dense(self.data_, (self.data_len_ + self.dense_len_)//2, tf.sigmoid)
        # self.dense_ = tf.layers.dense(self.layer_, self.dense_len_, tf.sigmoid)
        
        # self.dense_ = tf.layers.dense(self.data_, self.dense_len_, activation=tf.tanh)

        self.dense_ = tf.layers.dense(self.data_, self.dense_len_, activation=tf.math.atan , name = "fcencoder")
        # self.dense_ = tf.pow(self.dense_, 1/3)



class CnnEncoder(VisualEncoder):
    # [Assumption]
    # 1. with padding
    # 2. NHWC
    # 3. (possible to improve) last layer is fully connected with relu and stretched
    '''
    data: a tensor (placeholder)
    is_train: a bool tensor (placeholder)
    '''
    def __init__(self, data, dense_len, strides, height, width, channel, is_train):
        super().__init__(data, dense_len)
        self.strides_ = strides
        self.height_ = height
        self.width_ = width
        self.channel_ = channel
        self.is_train_ = is_train

        # with tf.variable_scope('CnnEncoder', reuse=False):
        self.is_train_ = is_train
        self.layers_ = [self.data_]
        for stride in self.strides_:
            conv_layer = tf.layers.conv2d(self.layers_[-1], 32, 3, strides=stride, padding='same', activation=None)
            conv_layer_norm = tf.layers.batch_normalization(conv_layer, training=self.is_train_)
            conv_layer_relu = tf.nn.relu(conv_layer_norm)
            self.layers_.append(conv_layer_relu)

        conv_layer_last = self.layers_[-1]
        feature_dim = conv_layer_last.shape[1] * conv_layer_last.shape[2] * conv_layer_last.shape[3]
        self.conv_layer_last_stretched_ = tf.reshape(conv_layer_last, [-1, feature_dim])
        self.dense_ = tf.layers.dense(self.conv_layer_last_stretched_, self.dense_len_, tf.nn.relu)


if __name__ == '__main__':
    np.random.seed(2018)
    symbolic_data = np.random.randint(2, size=[10, 500])
    visual_data = np.random.randint(256, size=[10, 124, 124, 3])
    fc = FcEncoder(50, 500)
    cnn = CnnEncoder(50, [2, 1, 1, 2, 1, 2, 1, 2], 124, 124)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run([fc.dense, cnn.dense], feed_dict={fc.data: symbolic_data, cnn.data: visual_data, cnn.is_train: True}))
