import tensorflow as tf
from VisualEncoder import ConvAsFcEncoder
# from vgg16 import vgg16
import numpy as np
class Agnostic_Speaker:

    @property
    def message(self):
        return self.message_

    @property
    def log_prob(self):
        return self.log_prob_

    def __init__(self, encode_type, input_len, dense_len, num_distract, vocabulary_size, temperature, img_height=None, img_width = None, sess = None, **kwargs):

        if encode_type == 'fc':
            self.target_ = tf.placeholder(dtype = tf.float32, shape = (None, input_len), name='target')
            self.ori_distract_ = tf.placeholder(dtype = tf.float32, shape = (None,  num_distract, input_len), name = 'distract')
            self.distract_ = tf.transpose(self.ori_distract_, perm = [0, 2, 1])
            
        elif encode_type == 'vgg':
            #first go through VGG
            assert(sess is not None and img_height is not None and img_width is not None) 
            self.target_imgs_ = tf.placeholder(dtype = tf.float32, shape =(None, img_height, img_width, 3))
            self.distract_imgs_ = tf.placeholder(dtype = tf.float32, shape =(None, img_height, img_width, 3))
            vgg_target_ = vgg16(self.target_imgs_, 'vgg16_weights.npz', sess)
            vgg_distract_ = vgg16(self.distract_imgs_, 'vgg16_weights.npz', sess)
            self.target_ = (np.argsort(vgg_target_.probs)[::-1])[0:input_len]
            self.distract_ = (np.argsort(vgg_distract_.probs)[::-1])[0:input_len]
       
        self.data_ = tf.expand_dims(tf.concat([tf.expand_dims(self.target_, axis=-1), self.distract_], axis=-1), axis = -1)
        #after expand_dim self.data_ should have shape batch_size * input_len * (num_distract+1) * 1
        self.data_encoder_ = ConvAsFcEncoder(self.data_, dense_len, (input_len, 1), dense_len, strides=(1,1), activation_fun = tf.sigmoid, name = "speaker_data_encoder")
        #after fully connected the desnse output should have size batch_size * 1 * (num_distract + 1) * dense_len
        #the dense should have shape batch_size * dense_len * (num_distract + 1) * 1
        self.symbols_ = ConvAsFcEncoder(tf.transpose(self.data_encoder_.dense, perm=[0,3,2,1]), dense_len, (dense_len, num_distract+1), vocabulary_size, strides=(1,1), name = "speaker_symbols").dense
        #after fully connected, the shape would be batch_size * 1 * 1 * vocabulary size
        numer = tf.exp(tf.negative(tf.squeeze(self.symbols_)) / temperature)
        denom = tf.reshape(tf.reduce_sum(numer, axis=1), (-1,1))
        self.probabilities_ = numer / denom
        self.distribution_ =  tf.distributions.Categorical(probs = self.probabilities_)
        sampled_idx = self.distribution_.sample()
        self.message_ = tf.one_hot(sampled_idx, vocabulary_size, dtype=tf.float32)
        self.log_prob_ = tf.log(tf.gather_nd(self.probabilities_, tf.stack([tf.range(tf.shape(self.probabilities_)[0]), sampled_idx], axis=1)))
        print("Speaker tensor: {}".format(self.log_prob_))
        
class Informed_Speaker:
    @property
    def message(self):
        return self.message_
    
    @property
    def log_prob(self):
        return self.log_prob_

    @property
    def logits(self):
        return self.logits_
    # @property
    # def reg_loss(self):
    #     return  self.regularization_
        
    def __init__(self, encode_type, input_len, dense_len, num_distract, num_filter, vocabulary_size, temperature, img_height = None, img_width = None, sess=None,**kwargs):
        if encode_type == 'fc':
            self.target_ = tf.placeholder(dtype = tf.float32, shape = (None, input_len), name = 'speaker_target')
            self.ori_distract_ = tf.placeholder(dtype = tf.float32, shape = (None,  num_distract, input_len), name = 'speaker_distract')
            self.distract_ = tf.transpose(self.ori_distract_, perm = [0, 2, 1])
        elif encode_type == 'vgg':
            #first go through VGG
            assert(sess is not None and img_height is not None and img_width is not None)            
            self.target_imgs_ = tf.placeholder(dtype = tf.float32, shape =(None, img_height, img_width, 3))
            self.distract_imgs_ = tf.placeholder(dtype = tf.float32, shape =(None, img_height, img_width, 3))
            vgg_target_ = vgg16(self.target_imgs_, 'vgg16_weights.npz', sess)
            vgg_distract_ = vgg16(self.distract_imgs_, 'vgg16_weights.npz', sess)
            self.target_ = (np.argsort(vgg_target_.probs)[::-1])[0:input_len]
            self.distract_ = (np.argsort(vgg_distract_.probs)[::-1])[0:input_len]
        
        #save the inputs for testing
        tf.get_default_graph().add_to_collection("Speaker_input", self.target_)
        tf.get_default_graph().add_to_collection("Speaker_input", self.ori_distract_)
 
        self.data_ = tf.expand_dims(tf.concat([tf.expand_dims(self.target_, axis=-1) , self.distract_], axis=-1), axis=-1)
        
        with tf.variable_scope('Teacher_Update'):
            #no sigmoid nonlinearlity here 
            self.data_encoder_ = ConvAsFcEncoder(self.data_, dense_len, (input_len, 1), dense_len, strides=(1,1), name = "speaker_data_encoder")
            #after the above fully connected operation, the output size is batch_size * 1*(num_distract+1)*dense_len
            #use transpose to change shape to batch_size * (num_distract + 1) * dense_len * 1
            self.feature_maps_ = tf.layers.conv2d(tf.transpose(self.data_encoder_.dense, perm=[0, 2, 3, 1]), filters = num_filter, kernel_size = (num_distract+1,1),strides=(1,1), activation = tf.sigmoid, name = 'feature_map')
            #after convolution, the shape would be batch_size * 1 * dense_len * num_filters
            #need to transpose to batch_size * num_filter * dense_len * 1
            self.combined_feature_map_ = tf.layers.conv2d(tf.transpose(self.feature_maps_, perm = [0, 3, 2, 1]), filters = 1, kernel_size = (num_filter, 1), strides = (1,1), name = 'combined_feature_map')
            #after combination,  the shape would be batch_size * 1 * dense_len * 1
            
            #the next step is not mentioned in the paper, need to find further confirmation
            self.logits_ = ConvAsFcEncoder(self.combined_feature_map_, dense_len, (1, dense_len), vocabulary_size, strides = (1,1), name = "speaker_symbols").dense
            self.logits_ = tf.squeeze(self.logits_)
            self.probabilities_ = tf.nn.softmax(self.logits_/temperature)
            # self.numer = tf.exp(self.logits_ / temperature)
            # self.denom = tf.reshape(tf.reduce_sum(self.numer, axis=1),(-1,1))
            # self.probabilities_ = self.numer / self.denom
            self.distribution_ =  tf.distributions.Categorical(probs = self.probabilities_)
            sampled_idx = self.distribution_.sample()
            self.message_ = tf.one_hot(sampled_idx, vocabulary_size, dtype=tf.float32, name = 'speaker_message')
            tf.get_default_graph().add_to_collection("Speaker_input", self.message_)

            self.log_prob_ = tf.log(tf.gather_nd(self.probabilities_, tf.stack([tf.range(tf.shape(self.probabilities_)[0]), sampled_idx], axis=1)))

            self.reg_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('Teacher')]
            # self.regularization_ = 0 * tf.add_n([ tf.nn.l2_loss(v) for v in self.reg_varlist_ if 'bias' not in v.name ])
            

            
