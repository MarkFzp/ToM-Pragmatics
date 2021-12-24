import sys
import os
import tensorflow as tf
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from vgg16 import Vgg16

class VisualEncoder:
    def __init__(self, img_h, img_w, inputs):
        self.image_h_ = img_h
        self.image_w_ = img_w
        self.inputs_ = inputs # [None, num_distractors, image_h, image_w, 3], rgb, 0-255
        self.num_distractors_ = self.inputs_.shape[1]
        self.inputs_reshape_ = tf.reshape(self.inputs_, [-1, self.image_h_, self.image_w_, 3])
    
    def build(self):
        for idx, out_dim in enumerate(self.fc_layer_info_):
            fc = tf.layers.conv2d(self.fc_layers_[-1], out_dim, kernel_size=1, strides=1, padding='VALID', 
                                  activation = tf.nn.leaky_relu if idx != len(self.fc_layer_info_) - 1 else tf.sigmoid, kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 5e-2))
            self.fc_layers_.append(fc)
        
        self.visual_features_ = self.fc_layers_[-1]

class VGG16Encoder(VisualEncoder):
    def __init__(self, inputs, config_dict):
        VisualEncoder.__init__(self, config_dict.img_h, config_dict.img_w, inputs)
        self.core_ = Vgg16(config_dict.vgg16_npy_path)
        self.image_features_ = self.core_.build(self.inputs_reshape_)
        
        self.fc_layer_info_ = config_dict.fc_configs

        self.fc_layers_ = [tf.reshape(self.image_features_, [-1, self.num_distractors_, 1, self.image_features_.shape[1]])]
        self.build()

class CNNEncoder(VisualEncoder):
    def __init__(self, inputs, config_dict):
        VisualEncoder.__init__(self, config_dict.img_h, config_dict.img_w, inputs)

        self.cnn_layer_info_ = config_dict.cnn_configs
        self.fc_layer_info_ = config_dict.fc_configs

        self.cnn_layers_ = [self.inputs_reshape_]
        for i, pair in enumerate(self.cnn_layer_info_):
            kernel_size, stride, out_layer, max_pool = pair
            conv_layer = tf.layers.conv2d(self.cnn_layers_[-1], out_layer, kernel_size, strides=stride, padding='SAME', 
                                        kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 5e-2), activation=None)
            if max_pool:
                pool_layer = tf.nn.max_pool(conv_layer, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME')
            else:
                pool_layer = tf.identity(conv_layer)
            self.cnn_layers_.append(tf.nn.leaky_relu(pool_layer))
        
        self.fc_layers_ = [tf.reshape(self.cnn_layers_[-1], [-1, self.num_distractors_, 1, np.product(self.cnn_layers_[-1].get_shape()[1:])])]
        self.build()
        

if __name__ == "__main__":
    main()