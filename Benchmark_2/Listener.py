import tensorflow as tf
from VisualEncoder import ConvAsFcEncoder
class Listener:
    
    '''
    return: 1d float32 tensor
    '''
    @property
    def index_match_indicator(self):
        return self.index_match_indicator_
    
    @property
    def log_prob(self):
        return self.log_prob_
    
    @property
    def logits(self):
        return self.logits_
    # @property
    # def reg_loss(self):
    #     return  self.regularization_

    def __init__(self, input_len, dense_len, num_candidates, message, vocabulary_size, temperature, **kwargs):
        self.is_train_ = tf.placeholder(dtype = tf.bool, shape = (), name = 'listener_is_train')
        self.ori_data_ = tf.placeholder(dtype = tf.float32, shape = (None, num_candidates, input_len), name = 'listener_data')
        self.attr_equiv_class_ = tf.placeholder(dtype = tf.int32, shape = (None, vocabulary_size))
        self.data_ = tf.transpose(self.ori_data_, perm=[0,2,1])
        self.target_idx_ = tf.placeholder(dtype=tf.int32, shape = (None), name = 'listener_target_idx')

        #save listener input
        tf.get_default_graph().add_to_collection("Listener_input", self.ori_data_)
        tf.get_default_graph().add_to_collection("Listener_input", self.target_idx_)
        tf.get_default_graph().add_to_collection("Listener_input", self.is_train_)
        tf.get_default_graph().add_to_collection("Listener_input", self.attr_equiv_class_)

        with tf.variable_scope('Student_Update'):
            self.data_encoder_ = ConvAsFcEncoder(tf.expand_dims(self.data_, axis=-1), dense_len, (input_len, 1), dense_len, strides = (1, 1), name = "student_data_encoder")
            #message is of shape batch_size * vocabulary_size
            self.message_ = tf.one_hot(tf.boolean_mask(self.attr_equiv_class_, tf.cast(tf.reshape(message, [-1,vocabulary_size]), dtype = tf.bool)), vocabulary_size, name = "listener_message")
            tf.get_default_graph().add_to_collection("Listener_input", self.message_)

            # self.message_ = tf.reshape(message, [-1,vocabulary_size])
            self.symbol_encoder_ = ConvAsFcEncoder(tf.expand_dims(tf.expand_dims(self.message_,axis=-1), axis=-1), dense_len, (vocabulary_size, 1), dense_len, strides = (1,1), name = "student_symbol_encoder")
            #symbol embedding has shape batch_size * 1 * 1 * dense_len
            #data embedding has shape batch_size * 1 * (num_distract + 1) * dense_len
            

            data_temp = tf.squeeze(tf.transpose(self.data_encoder_.dense, perm=[0,3,2,1]), axis=[-1])
            symbol_temp = tf.squeeze(tf.transpose(self.symbol_encoder_.dense, perm=[0,3,2,1]), axis=[-1])
            #rec_vec has shape batch_size * (num_distract + 1)
            self.logits_ = tf.reduce_sum(tf.multiply(data_temp, symbol_temp), axis=1, keep_dims=False)
            self.probabilities_ = tf.nn.softmax(self.logits_ / temperature)
            # numer = tf.exp(tf.negative(self.logits_) / temperature)
            # denom = tf.reshape(tf.reduce_sum(numer, axis=1), (-1,1))
            # self.probabilities_ = numer/denom
            self.distribution_ = tf.distributions.Categorical(probs= self.probabilities_)
            self.sampled_idx_ = tf.cond(self.is_train_, lambda: self.distribution_.sample(), lambda: tf.argmax(self.probabilities_, axis=1, output_type=tf.int32))
            # self.sampled_idx_ = self.distribution_.sample() 
            self.log_prob_ = tf.log(tf.gather_nd(self.probabilities_, tf.stack([tf.range(tf.shape(self.probabilities_)[0]), self.sampled_idx_], axis=1)))
            # print("Listener tensor: {}".format(self.log_prob_))
            self.index_match_indicator_ = tf.cast(tf.equal(self.target_idx_, self.sampled_idx_), dtype= tf.int32)

            self.reg_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('Student')]
            # self.regularization_ = 1e-6 * tf.add_n([ tf.nn.l2_loss(v) for v in self.reg_varlist_ if 'bias' not in v.name ])
