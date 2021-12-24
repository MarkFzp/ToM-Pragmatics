import tensorflow as tf
import numpy as np
from abc import ABC, abstractclassmethod

class Dense2Message(ABC):
    def __init__(self, dense_len, message_len, alphabet_size):
        self.hidden_dim_ = self.dense_len_ = dense_len
        self.message_len_ = message_len
        self.alphabet_extended_size_ = alphabet_size + 1
        self.message_ = None
        self.log_prob_sum_ = None
        self.entropy_ = None
        self.logits_ = None
    @property
    def message(self):
        assert(self.message_ is not None)
        return self.message_

    @property
    def log_prob_sum(self):
        assert(self.log_prob_sum_ is not None)
        return self.log_prob_sum_
    
    @property
    def entropy(self):
        assert(self.entropy_ is not None)
        return self.entropy_

    @property
    def logits(self):
        assert(self.logits_ is not None)
        return self.logits_
        


class LstmD2M(Dense2Message):
    '''
    alphabet_size: max number of symbols (excluding the end symbol)
    dense: a tensor (placeholder)
    is_train: a bool tensor (placeholder)
    '''
    def __init__(self, dense, dense_len, message_len, tempurature, alphabet_size, is_train):
        assert(type(alphabet_size) == int and alphabet_size > 0)
        super().__init__(dense_len, message_len, alphabet_size)
        self.tempurature_ = tempurature
        self.is_train_ = is_train

        with tf.variable_scope('LstmD2M'):
            if dense is not None:
                self.dense_ = dense
                self.dense_.set_shape([None, self.dense_len_])
            else:
                self.dense_ = tf.placeholder(tf.float32, [None, self.dense_len_])
            self.batch_size_ = tf.shape(self.dense_)[0]

            self.cell_ = tf.nn.rnn_cell.LSTMCell(self.hidden_dim_, state_is_tuple=True, name='cell_')
            # self.in_states_ = [(self.dense_, tf.zeros([self.batch_size_, self.hidden_dim_], name='in_states__zeros'))]
            self.in_states_ = [(self.dense_, tf.tanh(self.dense_))]
            self.in_chars_ = [tf.zeros([self.batch_size_, self.alphabet_extended_size_], name='in_chars__zeros')]
            self.output_W = tf.get_variable('output_W', [self.hidden_dim_, self.alphabet_extended_size_])
            self.output_b = tf.get_variable('output_b', [self.alphabet_extended_size_], initializer=tf.initializers.zeros)

            zeros = tf.zeros([self.batch_size_], name='zeros')
            minusone = tf.fill([self.batch_size_], -1, name='minusone')
            self.log_prob_sum_ = self.entropy_ = zeros
            self.message_ = None
            self.is_not_end_ = tf.fill([self.batch_size_], True)

            ### for only 1 run of cell ###
            # in_state = self.in_states_[-1]
            # in_char = self.in_chars_[-1]
            # h, out_state = self.cell_(in_char, in_state)
            # out_prob = tf.add(tf.matmul(h, self.output_W, name='matmul'), self.output_b, name='out_prob')
            # prob_before_norm = tf.exp(out_prob / self.tempurature_, name='exp')
            # prob_before_norm = tf.concat([prob_before_norm[:, :-1], tf.zeros([self.batch_size_, 1])], axis=1, name='concat')
            # prob = tf.truediv(prob_before_norm, tf.reshape(tf.reduce_sum(prob_before_norm, 1, name='reduce_sum'), [-1, 1], name='reshape'), name='prob')
            # prob_non_zero = prob[:, :-1]
            # self.entropy_ = - tf.reduce_sum(prob_non_zero * tf.log(prob_non_zero, name='log'), 1, name='prob_entropy')
            # sample_index = tf.cond(self.is_train_, lambda: tf.reshape(tf.multinomial(prob, num_samples=1, output_dtype=tf.int32, name='multinomial'), [-1], name='sample_index_cond0'), lambda: tf.argmax(prob, axis=1, output_type=tf.int32, name='argmax'))
            # self.log_prob_sum_ = tf.log(tf.gather_nd(prob, tf.stack([tf.range(self.batch_size_), sample_index], axis=1, name='stack'), name='gather_nd') + tf.fill([self.batch_size_], 1e-12), name='sample_log_prob')
            # self.message_ = tf.reshape(sample_index, [-1, 1])
            
            self.h = self.out_prob = self.prob = []
            
            #currently message_len_ = 1
            for i in range(self.message_len_):
                in_state = self.in_states_[-1]
                in_char = self.in_chars_[-1]
                h, out_state = self.cell_(in_char, in_state)
                self.h.append(h)
                out_prob = tf.add(tf.matmul(h, self.output_W, name='matmul_{}'.format(i)), self.output_b, name='out_prob_{}'.format(i))
                self.out_prob.append(out_prob)
                
                # first symbol cannnot be stop symbol
                if i == 0:
                    out_prob = out_prob[:, :-1]
                
                self.logits_ = out_prob
                prob = tf.nn.softmax(out_prob / self.tempurature_)
                self.prob.append(prob)
                prob_entropy_ = - tf.reduce_sum(prob * tf.log(1e-10 + prob, name='log_{}'.format(i)), 1, name='prob_entropy_{}'.format(i))
                
                sample_index = tf.cond(self.is_train_, lambda: tf.distributions.Categorical(probs=prob).sample(1)[0], lambda: tf.argmax(prob, axis=1, output_type=tf.int32, name='argmax_{}'.format(i)))
                sample_log_prob = tf.log(1e-10 + tf.gather_nd(prob, tf.stack([tf.range(self.batch_size_, name='range_{}'.format(i)), sample_index], axis=1, name='stack_{}'.format(i)), name='gather_nd_{}'.format(i)), name='sample_log_prob_{}'.format(i))

                # log_prob_sum includes the prob of sampling out end_char
                self.log_prob_sum_ += tf.where(self.is_not_end_, sample_log_prob, zeros)
                self.entropy_ += tf.where(self.is_not_end_, prob_entropy_, zeros)
                self.is_not_end_ = tf.logical_and(self.is_not_end_, tf.logical_not(tf.equal(sample_index, self.alphabet_extended_size_ - 1, name='is_not_end_{}'.format(i))))
                current_iter_message = tf.reshape(tf.where(self.is_not_end_, sample_index, minusone), [-1, 1])
                if self.message_ is None:
                    self.message_ = current_iter_message
                else:
                    self.message_ = tf.concat([self.message_, current_iter_message], axis=1)

                self.in_states_.append((out_state.c, out_state.h))
                self.in_chars_.append(tf.one_hot(sample_index, depth=self.alphabet_extended_size_, name='one_hot_{}'.format(i)))


if __name__ == '__main__':
    np.random.seed(100)
    np.set_printoptions(threshold=np.nan)
    is_train = tf.placeholder(tf.bool, [])
    dense = tf.placeholder(tf.float32, [2, 50])
    d2m = LstmD2M(dense, 50, 5, 1.5, 5, is_train)

    writer = tf.summary.FileWriter('./graph_placeholder', tf.get_default_graph())
    writer.close()

    with tf.Session() as sess:
        fd = {is_train: True, dense: np.random.normal(size=[2, 50])}
        tf.global_variables_initializer().run()
        print(sess.run([d2m.message, d2m.log_prob_sum, d2m.entropy], feed_dict=fd))
