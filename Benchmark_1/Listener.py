from VisualEncoder import FcEncoder, CnnEncoder
from Message2Dense import LstmM2D
import tensorflow as tf

class Listener:
    
    '''
    for every property
    return: 1d tensor
    '''

    @property
    def log_prob(self):
        return self.sample_log_prob_

    '''
    return: 1d float32 tensor
    '''
    @property
    def match_indicator(self):
        return self.match_indicator_
    
    @property
    def entropy(self):
        return self.entropy_


    '''
    candidate_count: number of distractors per target + 1 (the target)
    '''
    def __init__(self, encoder_type, data_len, dense_len, candidate_count, message, max_message_len, alphabet_size, temperature, **kwargs):
        ###########################
        self.target_index = tf.placeholder(tf.int32, [None], name='listener_target_index')
        self.is_train = tf.placeholder(tf.bool, [], name='listener_is_train')
        self.attr_equiv_class_ = tf.placeholder(dtype = tf.int32, shape = (None, alphabet_size))

        ###########################
        assert(encoder_type in ['fc', 'cnn'])
        self.encoder_type_ = encoder_type
        self.data_len_ = data_len
        self.dense_len_ = dense_len
        self.candidate_count_ = candidate_count
        # self.message_ = message
        idx = tf.range(tf.shape(message)[0])
        message = tf.reshape(message, [-1])
        self.message_ = tf.reshape(tf.gather_nd(self.attr_equiv_class_,tf.stack([idx, message], axis = 1)), [-1,1] )
        # self.message_ = tf.boolean_mask(self.attr_equiv_class_, tf.cast(tf.reshape(message, [-1,alphabet_size]), dtype = tf.bool))

        self.max_message_len_ = max_message_len
        self.alphabet_size_ = alphabet_size
        self.temperature_ = temperature

        with tf.variable_scope("Student_Update"):
            if self.encoder_type_ == 'fc':
                ###########################
                self.data = tf.placeholder(tf.float32, [None, data_len], name='listener_data')
                ###########################
                self.data_encoder_ = FcEncoder(self.data, self.dense_len_, self.data_len_)
            elif self.encoder_type_ == 'cnn':
                assert(kwargs is not None)
                keys = list(kwargs)
                for key in ['strides', 'height', 'width', 'channel']:
                    assert(key in keys)
                    setattr(self, key + '_', kwargs[key])
                ###########################
                self.data = tf.placeholder(tf.float32, [None, self.height_, self.width_, self.channel_], name='listener_data')
                ###########################
                self.data_encoder_ = CnnEncoder(self.data, self.dense_len_, self.strides_, self.height_, self.width_, self.channel_, self.is_train)
            
            self.distractor_dense_ = self.data_encoder_.dense
            self.m2d_ = LstmM2D(self.message_, self.max_message_len_, self.dense_len_, self.alphabet_size_)
            self.message_dense_ = self.m2d_.dense
            message_dense_stack = tf.reshape(tf.concat([self.message_dense_] * self.candidate_count_, 1), [-1, self.dense_len_])
            self.dot_product_ = tf.reshape(tf.reduce_sum(self.distractor_dense_ * message_dense_stack, 1), [-1, self.candidate_count_])

            prob = tf.nn.softmax(self.dot_product_ / temperature)

            # prob_before_norm = tf.exp(self.dot_product_ / self.temperature_)
            # prob = prob_before_norm / tf.reshape(tf.reduce_sum(prob_before_norm, 1), [-1, 1])
            self.chosen_index_ = tf.cond(self.is_train, lambda: tf.distributions.Categorical(probs=prob).sample(1)[0], lambda: tf.argmax(prob, axis=1, output_type=tf.int32))
            self.match_indicator_ = tf.cast(tf.equal(self.chosen_index_, self.target_index), tf.float32)
            self.sample_log_prob_ = tf.log(1e-10 + tf.gather_nd(prob, tf.stack([tf.range(tf.shape(self.chosen_index_)[0]), self.chosen_index_], axis=1)))
            self.entropy_ = - tf.reduce_sum(prob * tf.log(1e-10 + prob), axis=1)
