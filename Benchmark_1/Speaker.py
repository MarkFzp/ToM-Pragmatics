from VisualEncoder import FcEncoder, CnnEncoder
from Dense2Message import LstmD2M
import tensorflow as tf

class Speaker:

    @property
    def message(self):
        return self.message_

    @property
    def log_prob_sum(self):
        return self.log_prob_sum_
    
    @property
    def entropy(self):
        return self.entropy_

    @property
    def logits(self):
        return self.logits_

    def __init__(self, encoder_type, data_len, dense_len, message_len, alphabet_size, temperature, **kwargs):
        assert(encoder_type in ['fc', 'cnn'])
        self.encoder_type_ = encoder_type
        self.data_len_ = data_len
        self.dense_len_ = dense_len
        self.message_len_ = message_len
        self.alphabet_size_ = alphabet_size
        self.temperature_ = temperature
        ###########################
        self.is_train = tf.placeholder(tf.bool, [], name='speaker_is_train')
        ###########################
        with tf.variable_scope("Teacher_Update"):
            if self.encoder_type_ == 'fc':
                ###########################
                self.target = tf.placeholder(tf.float32, [None, self.data_len_], name='speaker_target')
                ###########################
                self.data_encoder_ = FcEncoder(self.target, self.dense_len_, self.data_len_)
            elif self.encoder_type_ == 'cnn':
                assert(kwargs is not None)
                keys = list(kwargs)
                for key in ['strides', 'height', 'width', 'channel']:
                    assert(key in keys)
                    setattr(self, key + '_', kwargs[key])
                ###########################
                self.target = tf.placeholder(tf.float32, [None, self.height_, self.width_, self.channel_], name='speaker_target')
                ###########################
                self.data_encoder_ = CnnEncoder(self.target, self.dense_len_, self.strides_, self.height_, self.width_, self.channel_, self.is_train)
            
            self.dense_ = self.data_encoder_.dense
            self.d2m_ = LstmD2M(self.dense_, self.dense_len_, self.message_len_, self.temperature_, self.alphabet_size_, self.is_train)
            self.message_ = self.d2m_.message
            self.log_prob_sum_ = self.d2m_.log_prob_sum
            self.entropy_ = self.d2m_.entropy
            self.logits_ = self.d2m_.logits_
