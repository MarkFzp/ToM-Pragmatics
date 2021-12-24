import numpy as np
import tensorflow as tf

from base import Model
from belief_update import Belief_Update

class Belief_Q(Model):
	def __init__(self):
		def __init__(self, sess, num_distractors, attributes_size,
					 message_space_size, learning_rate):
		self.sess = sess
		self.tf_belief_update_net = Belief_Update(num_distractors, attributes_size, message_space_size)
		self.num_distractors = num_distractors
		self.attributes_size = attributes_size
		self.message_space_size = message_space_size
		self.learning_rate = learning_rate
		#self.opt = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)
		#self.opt = tf.train.MomentumOptimizer(learning_rate = self.learning_rate, momentum = 0.9)
		self.opt = tf.train.AdamOptimizer()

		self.belief_self = tf.placeholder(tf.float32, shape = [None, self.num_distractors], name = 'belief_self')
		self.distractors = tf.placeholder(tf.float32, name = 'distractors',
											  shape = [None, self.num_distractors, self.attributes_size])
		self.distractors_tensor = tf.expand_dims(self.distractors, 2)
		self.df1 = tf.layers.conv2d(self.distractors_tensor, 3 * self.message_space_size, kernel_size = [1, 1],
							   		kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2),
							   		activation = tf.nn.leaky_relu)
		self.df2 = tf.layers.conv2d(self.df1, 2 * self.message_space_size, kernel_size = [1, 1],
							   		kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2),
							   		activation = tf.nn.leaky_relu)
		self.df3 = tf.layers.conv2d(self.df2, 1 * self.message_space_size, kernel_size = [1, 1],
							   		kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2),
							   		activation = None)

		self.df_b1 = tf.multiply(tf.squeeze(self.df3), tf.expand_dims(self.belief_self, -1))
		self.df_b2 = tf.multiply(tf.squeeze(self.df3), tf.expand_dims(self.tf_belief_update_net, -1))
		self.value = tf.reduce_sum(tf.math.square(self.df_b1 - self.df_b2))

if __name__ == '__main__':
	main()