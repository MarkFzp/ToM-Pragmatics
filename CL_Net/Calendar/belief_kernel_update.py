import numpy as np
import tensorflow as tf

from base import Model
from concept import Concept

import pdb

def tf_JS_divergence(P, Q):
	M = 0.5 * (P + Q)
	KL1 = tf_KL_divergence(P, M)
	KL2 = tf_KL_divergence(Q, M)
	return 0.5 * (KL1 + KL2)

def tf_KL_divergence(P, Q):
	X = tf.distributions.Categorical(probs = P)
	Y = tf.distributions.Categorical(probs = Q)
	return tf.distributions.kl_divergence(X, Y)

class Belief_Update(Model):
	def __init__(self, sess, num_distractors, attributes_size, message_space_size, learning_rate):
		self.sess = sess
		self.num_distractors = num_distractors
		self.attributes_size = attributes_size
		self.message_space_size = message_space_size
		self.learning_rate = learning_rate
		#self.opt = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)
		#self.opt = tf.train.MomentumOptimizer(learning_rate = self.learning_rate, momentum = 0.9)
		self.opt = tf.train.AdamOptimizer()

		with tf.variable_scope('belief_kernel_update'):
			self.prev_belief = tf.placeholder(tf.float32, shape = [None, self.num_distractors], name = 'prev_belief')
			self.message = tf.placeholder(tf.float32, shape = [None, self.message_space_size], name = 'message')
			self.distractors = tf.placeholder(tf.float32, name = 'distractors',
											  shape = [None, self.num_distractors, self.attributes_size])
			self.new_belief = tf.placeholder(tf.float32, shape = [None, self.num_distractors], name = 'new_belief')

			self.kernel_columns = []
			for i in range(self.num_distractors):
				self.distractors_tensor = tf.expand_dims(self.distractors, 2)
				self.df1 = tf.layers.conv2d(self.distractors_tensor, 3 * self.message_space_size, kernel_size = [1, 1],
									   		kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2),
									   		activation = tf.nn.leaky_relu)
				self.df2 = tf.layers.conv2d(self.df1, 2 * self.message_space_size, kernel_size = [1, 1],
									   		kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2),
									   		activation = tf.nn.leaky_relu)
				self.df3 = tf.layers.conv2d(self.df2, 1 * self.message_space_size, kernel_size = [1, 1],
									   		kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2),
									   		activation = tf.nn.leaky_relu)
				#df_reshape = tf.reshape(self.df3, shape = [-1, self.num_distractors * 1 * self.message_space_size])
				#self.df_msg_1 = tf.concat([df_reshape, self.message], axis = 1)
				

				# self.df_msg_1 = tf.reshape(tf.multiply(self.df2, tf.expand_dims(tf.expand_dims(self.message, 1), 1)),
				# 						   shape = [-1, self.num_distractors * self.message_space_size])
				# self.df_msg_2 = tf.contrib.layers.fully_connected(self.df_msg_1, 2 * self.num_distractors, activation_fn = tf.nn.leaky_relu)
				# self.df_msg_3 = tf.contrib.layers.fully_connected(self.df_msg_2, 1 * self.num_distractors, activation_fn = tf.nn.sigmoid)
				# kernel_column = tf.contrib.layers.fully_connected(self.df_msg_3,
				# 												  self.num_distractors,
				# 												  activation_fn = tf.nn.sigmoid) + 1e-6
				self.df_msg_1 = tf.multiply(self.df3, tf.expand_dims(tf.expand_dims(self.message, 1), 1))
				self.df_msg_2 = tf.contrib.layers.fully_connected(tf.layers.flatten(self.df_msg_1),\
																  2 * self.num_distractors, activation_fn = tf.nn.leaky_relu)
				self.df_msg_3 = tf.contrib.layers.fully_connected(self.df_msg_2,
																  self.num_distractors, activation_fn = None)
				kernel_column = tf.math.exp(self.df_msg_3) + 1e-9
				#kernel_column = tf.math.exp(tf.reduce_sum(self.df_msg_1, axis = [2, 3]))
				self.kernel_columns.append(tf.expand_dims(tf.div(kernel_column,
														  tf.reduce_sum(kernel_column, axis = 1, keepdims = True)), -1))
			self.kernel = tf.concat(self.kernel_columns, axis = 2)
			print('<Belief Update Kernel Generator Constructed>')
			self.belief_pred = tf.squeeze(tf.matmul(self.kernel, tf.expand_dims(self.prev_belief, -1)))
			self.new_belief_smooth = tf.div(self.new_belief + 1e-6, tf.reduce_sum(self.new_belief + 1e-6))
			
			self.KL_divergence = tf.reduce_mean(tf_KL_divergence(self.new_belief_smooth, self.belief_pred))
						#-1 * tf.reduce_mean(tf.reduce_sum(tf.multiply(self.new_belief, tf.math.log(self.belief_pred)), axis = 1))
						#tf.reduce_mean(tf.reduce_sum(tf.multiply(self.belief_pred, tf.math.log(self.new_belief_smooth)), axis = 1))
			self.cross_entropy = -1 * tf.reduce_mean(tf.reduce_sum(tf.multiply(self.new_belief, tf.math.log(self.belief_pred)), axis = 1))
			self.JS_divergence = tf.reduce_mean(tf_JS_divergence(self.new_belief_smooth, self.belief_pred))
			self.mse = tf.reduce_mean(tf.reduce_sum(tf.square(self.belief_pred - self.new_belief), axis = 1))
			self.train_op = self.opt.minimize(self.JS_divergence)

	def train_step(self, data_batch):
		_, KL_divergence, JS_divergence, cross_entropy, belief_pred, kernel = self.sess.run([self.train_op, self.KL_divergence, self.JS_divergence, self.cross_entropy, self.belief_pred, self.kernel],
								feed_dict = {self.prev_belief: data_batch['prev_belief'],
											 self.message: data_batch['message'],
											 self.distractors: data_batch['distractors'],
											 self.new_belief: data_batch['new_belief']})
		#print(kernel)
		if np.isnan(cross_entropy):
			pdb.set_trace()
		return KL_divergence, JS_divergence, cross_entropy, belief_pred, kernel

def main():
	attributes_size = 10
	num_distractors = 4
	concept_max_size = 4
	attributes = range(attributes_size)

	training_steps = 100000
	batch_size = 128

	concept_generator = Concept(attributes, num_distractors, concept_max_size)
	
	sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, 
                                              log_device_placement = False))
	with tf.device('/gpu:0'):
		bu = Belief_Update(sess, 4, 10, 10, 1e-3)

	init = tf.global_variables_initializer()
	sess.run(init)
	
	for ts in range(training_steps):
		data_batch = concept_generator.generate_batch(batch_size)
		KL_divergence, JS_divergence, cross_entropy, belief_pred, kernel = bu.train_step(data_batch)
		if ts % 100 == 0:
			print('[%d] KL_divergence: %f, batch mean JS divergence: %f, batch mean cross entropy: %f'\
				  % (ts + 1, KL_divergence, JS_divergence, cross_entropy))
			idx = np.random.randint(batch_size)
			print('\t target:', data_batch['new_belief'][idx, :])
			print('\t predict', belief_pred[idx, :])

if __name__ == '__main__':
	main()