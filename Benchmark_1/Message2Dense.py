import tensorflow as tf
import numpy as np
from abc import ABC, abstractclassmethod

class Message2Dense:
    def __init__(self, dense_len, max_message_len, alphabet_size):
        self.dense_len_ = self.hidden_dim_ = dense_len
        self.max_message_len_ = max_message_len
        self.alphabet_size_ = alphabet_size
        self.dense_ = None
    
    @property
    def dense(self):
        assert(self.dense_ is not None)
        return self.dense_


class LstmM2D(Message2Dense):
    def __init__(self, message, max_message_len, dense_len, alphabet_size=100):
        assert(type(dense_len) == int and dense_len > 0)
        assert(type(alphabet_size) == int and alphabet_size > 0)
        assert(type(max_message_len) == int and alphabet_size > 0)
        super().__init__(dense_len, max_message_len, alphabet_size)
        
        with tf.variable_scope('LstmM2D'):
            if message is not None:
                self.message_ = message
                self.message_.set_shape([None, self.max_message_len_])
            else:
                self.message_ = tf.placeholder(tf.int32, [None, self.max_message_len_])
            self.batch_size_ = tf.shape(self.message_)[0]
            is_stop_symbol = tf.equal(self.message_, -1)
            self.message_len_ = self.max_message_len_ - tf.reduce_sum(tf.cast(is_stop_symbol, tf.int32), axis=1)
            self.onehot_ = tf.one_hot(self.message_, self.alphabet_size_)
            # self.message_len_ = tf.reduce_sum(tf.reduce_max(tf.sign(tf.cast(self.onehot_, tf.int32)), 2), 1)
            # self.equal_ = tf.equal(self.message_len_ , self.message_len_2_)
            
            self.cell_ = tf.nn.rnn_cell.LSTMCell(self.hidden_dim_)
            zero_state = self.cell_.zero_state(self.batch_size_, dtype=tf.float32)
            _, out_state = tf.nn.dynamic_rnn(self.cell_, self.onehot_, self.message_len_, initial_state=zero_state)
            self.dense_ = out_state.h


if __name__ == "__main__":
    message = np.array([[60, 27, 14, 67, 95, 44, 69, 85, 50, -1],
                        [78, 40, 22, 33, 39, 59, 11, -1, -1, -1],
                        [90, 89, 85, 12, 32,  6, 18, 78, 44, 83],
                        [48, 35,  7, 37, 98, 57, 45, 66,  8, 28],
                        [ 7, 74, 31, 88, 11, -1, -1, -1, -1, -1],
                        [ 1, 32, 46,  3, 12,  1, 49,  5, 26, 93],
                        [88, 44, 78, 64, 47, 82, 59, 29, -1, -1],
                        [97, 49, 25, 75, 10, 57, 20, -1, -1, -1],
                        [52, 34, 48, 81, 33, 67, 38, 22, 67, 41],
                        [95, 32, 70, 13, 69, 88, 88, 52, 41, 81]])
    m2d = LstmM2D(None, 10, 50)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        dense = sess.run([m2d.dense, m2d.message_len_], feed_dict={m2d.message_: message})
        print(dense)
