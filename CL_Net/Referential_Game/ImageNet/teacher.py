import os
import math
import numpy as np
import tensorflow as tf

from vgg16 import Vgg16
from concept import Concept
import pdb

class Teacher:
	def __init__(self, sess, rl_gamma, boltzman_beta,
				 belief_var_1d, num_distractors, attributes_size,
				 message_space_size, img_length):
		self.sess = sess
		self.num_distractors_ = num_distractors
		self.attributes_size_ = attributes_size
		self.message_space_size_ = message_space_size
		self.rl_gamma_ = rl_gamma
		self.boltzman_beta_ = boltzman_beta
		self.belief_var_1d_ = belief_var_1d
		self.img_length = img_length
		################
		# Placeholders #
		################
		with tf.variable_scope('Teacher'):
			self.distractors_ = tf.placeholder(tf.float32, name = 'distractors', 
											  shape = [None, self.num_distractors_, self.img_length, self.img_length, 3])
			self.distractors_tensor_ = tf.reshape(self.distractors_, [-1, self.img_length, self.img_length, 3])
			self.message_ = tf.placeholder(tf.float32, shape = [None, self.message_space_size_], name = 'message')

			self.teacher_belief_ = tf.placeholder(tf.float32, shape = [None, self.num_distractors_], name = 'teacher_belief')
			self.student_belief_ = tf.placeholder(tf.float32, shape = [None, self.num_distractors_], name = 'student_belief')
			self.student_belief_spvs_ = tf.placeholder(tf.float32, shape = [None, self.num_distractors_], name = 'student_belief_spvs')
			
			self.q_net_spvs_ = tf.placeholder(tf.float32, shape = [None])
			# self.is_train = tf.placeholder(tf.bool, shape = [])
		########################
		# Belief Update Module #
		########################
		self.global_step = tf.Variable(0, trainable=False)
		self.starter_learning_rate = 1e-3
		self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step, 20000, 0.5, staircase=True)
		self.feature_update_opt_ = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
		with tf.variable_scope('Teacher_Feature_Extract'):
			self.perceptor_ = Vgg16()
			self.image_features_ = self.perceptor_.build(self.distractors_tensor_)
			#self.feature_extracted_pre_1_ = tf.contrib.layers.fully_connected(self.image_features_, 1024, activation_fn=tf.nn.leaky_relu)
			self.feature_extracted_ = tf.reshape(self.image_features_, [-1, self.num_distractors_, 1, np.product(self.image_features_.get_shape()[1:])])	
			print(self.feature_extracted_.shape)
		self.feature_train_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('Teacher_Feature_Extract')]
		self.feature_extract_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name.startswith('Teacher_Feature_Extract')]

		self.belief_update_opt_ = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
		with tf.variable_scope('Belief_Update'):
			self.df2_ = self.feature_extracted_
			# self.df1_ = tf.layers.conv2d(self.feature_extracted_, 3 * self.message_space_size_, kernel_size = [1, 1],
			# 					   		kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-1),
			# 					   		activation = tf.nn.leaky_relu)
			# self.df2_ = tf.layers.conv2d(self.df1_, 2 * self.message_space_size_, kernel_size = [1, 1],
			# 					   		kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-1),
			# 					   		activation = tf.nn.leaky_relu)
			# self.df3_ = tf.layers.conv2d(self.df2_, 1 * self.message_space_size_, kernel_size = [1, 1],
			# 					   		kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-1),
			# 					   		activation = None)

			self.msg_from_df_1_ = []
			for _ in range(self.num_distractors_):
				self.msg_from_df_1_.append(tf.layers.conv2d(self.df2_, 32, kernel_size = [self.num_distractors_, 1],
														  kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-1),
														  padding = 'valid', activation = tf.nn.leaky_relu))

			self.msg_est_tensor_1_ = tf.concat(self.msg_from_df_1_, axis = 1)

			self.msg_from_df_2_ = []
			for _ in range(self.num_distractors_):
				self.msg_from_df_2_.append(tf.layers.conv2d(self.msg_est_tensor_1_, 10, kernel_size = [self.num_distractors_, 1],
														  kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-1),
														  padding = 'valid', activation = tf.nn.leaky_relu))

			self.msg_est_tensor_2_ = tf.concat(self.msg_from_df_2_, axis = 1)
			
			self.reg_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('Belief') or v.name.startswith('Teacher_Feature_Extract')]
			#######################
			#network belief update#
			#######################
			self.msg_est_tensor_2d_ = tf.squeeze(self.msg_est_tensor_2_, axis = 2, name = "pre_softmax")
			
			self.belief_var_1d_ = tf.exp(tf.Variable(initial_value = self.belief_var_1d_, trainable = True, dtype = tf.float32))

			self.boltzman_beta_ = tf.Variable(initial_value = self.boltzman_beta_, trainable = False, dtype = tf.float32, name = 'boltzman_beta')

			
			self.msg_indices_ = tf.where(tf.not_equal(self.message_, 0))

			self.word_embedding_ = tf.get_variable(shape = [10, self.message_space_size_], name = 'word_embedding',
												   initializer = tf.initializers.random_normal(mean = 0, stddev = 1e-2))

			self.msg_embeddings_ = tf.expand_dims(tf.transpose(tf.gather(self.word_embedding_, self.msg_indices_[:, 1], axis = 1)), 1)
	
			self.df_msg_2_norm_ = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(self.msg_embeddings_, self.msg_est_tensor_2d_), axis = 2))
			self.belief_pred_1_ = tf.multiply(self.df_msg_2_norm_, self.student_belief_)
			self.belief_pred_full_ = tf.concat([self.belief_pred_1_, self.belief_var_1d_ * tf.slice(tf.ones_like(self.belief_pred_1_), [0, 0], [-1, 1])], axis = 1)
			
			self.belief_pred_full_norm_ = tf.div_no_nan(self.belief_pred_full_, tf.reduce_sum(self.belief_pred_full_, axis = 1, keepdims = True))
			self.belief_pred_ = tf.slice(self.belief_pred_full_norm_, [0, 0], [-1, self.num_distractors_])
			self.regularization_ = 1e-3 * tf.add_n([ tf.nn.l2_loss(v) for v in self.reg_varlist_ if 'bias' not in v.name ])
			
			self.cross_entropy_1_ = -1 * tf.reduce_mean(tf.reduce_sum(tf.multiply(self.student_belief_spvs_, tf.math.log(self.belief_pred_ + 1e-9)), axis = 1))
			self.cross_entropy_2_ = -1 * tf.reduce_mean(tf.reduce_sum(tf.multiply(self.belief_pred_, tf.math.log(self.student_belief_spvs_ + 1e-9)), axis = 1))
			self.cross_entropy_ = self.cross_entropy_1_ + self.cross_entropy_2_
			
			# self.cross_entropy_1_ = -1 * tf.reduce_sum(tf.multiply(self.student_belief_spvs_, tf.math.log(self.belief_pred_ + 1e-9)), axis = 1)
			# self.cross_entropy_2_ = -1 * tf.reduce_sum(tf.multiply(self.belief_pred_, tf.math.log(self.student_belief_spvs_ + 1e-9)), axis = 1)
			# self.cross_entropy_ = tf.reduce_mean(tf.cast(tf.count_nonzero(self.student_belief_spvs_, axis = 1) / 4 + 1, tf.float32) * (self.cross_entropy_1_ + self.cross_entropy_2_))
		
		self.belief_train_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('Belief_Update')]
		self.belief_update_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name.startswith('Belief_Update')]
		self.belief_update_train_op_ = self.belief_update_opt_.minimize(self.cross_entropy_, var_list = self.belief_train_varlist_, global_step=self.global_step)
		self.feature_update_train_op_ = self.feature_update_opt_.minimize(self.cross_entropy_, var_list = self.feature_train_varlist_)
		# self.feature_belief_update_train_op_ = self.belief_update_opt_.minimize(self.cross_entropy_)

		self.feature_extract_saver_ = tf.train.Saver()
		self.feature_extract_loader_ = tf.train.Saver(self.feature_extract_varlist_)
		self.belief_update_saver_ = tf.train.Saver()
		self.belief_update_loader_ = tf.train.Saver(self.belief_update_varlist_)
		self.feature_belief_saver_ = tf.train.Saver()
		self.feature_belief_loader_ = tf.train.Saver(self.feature_extract_varlist_ + self.belief_update_varlist_)
		# print(self.feature_train_varlist_)
		# print(self.belief_train_varlist_)
		####################
		# Q-network Module #
		####################
		self.q_net_opt_ = tf.train.AdamOptimizer(learning_rate = 1e-5)
		with tf.variable_scope('q_net'):
			self.distct_feat_1_ = tf.layers.conv2d(self.feature_extracted_, 3 * self.message_space_size_, kernel_size = [1, 1],
								   				   kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-1),
								   				   activation = tf.nn.leaky_relu)
			self.distct_feat_2_ = tf.layers.conv2d(self.distct_feat_1_, 2 * self.message_space_size_, kernel_size = [1, 1],
								   				   kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-1),
												   activation = tf.nn.leaky_relu)
			
			self.distct_feat_2_weighted_ = tf.multiply(self.distct_feat_2_, tf.expand_dims(tf.expand_dims(self.belief_pred_, -1), -1))
			
			self.distcts_feat_1_ = []
			for _ in range(self.num_distractors_):
				self.distcts_feat_1_.append(tf.layers.conv2d(self.distct_feat_2_weighted_, 1 * self.message_space_size_, kernel_size = [self.num_distractors_, 1],
														  kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-1),
														  padding = 'valid', activation = tf.nn.leaky_relu))

			self.distcts_feat_tensor_1_ = tf.concat(self.distcts_feat_1_, axis = 1)

			self.distcts_feat_2_ = []
			for _ in range(self.num_distractors_):
				self.distcts_feat_2_.append(tf.layers.conv2d(self.distcts_feat_tensor_1_, 1 * self.message_space_size_, kernel_size = [self.num_distractors_, 1],
														  kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-1),
														  padding = 'valid', activation = None))

			self.distcts_feat_tensor_2_ = tf.concat(self.distcts_feat_2_, axis = 1)

			self.custome_activaiton_ = lambda x: tf.where(tf.math.greater(x, 0), (tf.exp(x) - 1), (-1 * tf.exp(-x) + 1))
			self.distcts_feat_3_ = []
			for _ in range(self.num_distractors_):
				self.distcts_feat_3_.append(tf.layers.conv2d(self.distcts_feat_tensor_2_, 1, kernel_size = [self.num_distractors_, 1],
														  kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-1),
														  padding = 'valid', activation = self.custome_activaiton_))

			self.distcts_feat_tensor_3_ = tf.concat(self.distcts_feat_3_, axis = 1)
			
			self.value_param_1_ = tf.Variable(initial_value = -1, trainable = False, dtype = tf.float32)

			self.value_ = tf.reduce_sum(tf.multiply(tf.squeeze(self.distcts_feat_tensor_3_), self.teacher_belief_), axis = 1) +\
						  (1 - tf.reduce_sum(self.belief_pred_, axis = 1)) * self.value_param_1_

			self.reg_varlist_q_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('q_net')]
			self.regularization_q_ = 1e-4 * tf.add_n([ tf.nn.l2_loss(v) for v in self.reg_varlist_q_ if 'bias' not in v.name ])
			self.q_net_loss_pre_ = tf.square(self.value_ - self.q_net_spvs_)
			self.success_mask_ = tf.to_float(tf.math.greater(self.q_net_spvs_, 0.0))
			self.fail_mask_ = tf.to_float(tf.math.greater(0.0, self.q_net_spvs_))
			self.imbalance_penalty_ = self.success_mask_ + self.fail_mask_ * tf.div_no_nan(tf.reduce_sum(self.success_mask_), tf.reduce_sum(self.fail_mask_))
			# self.q_net_loss_ = tf.reduce_mean(self.q_net_loss_pre_ * tf.to_float(self.q_net_loss_pre_ > 0.05) * self.imbalance_penalty_) + self.regularization_q_
			self.q_net_loss_ = tf.reduce_mean(self.q_net_loss_pre_ * self.imbalance_penalty_) + self.regularization_q_
			self.q_net_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('q_net')]
			self.q_net_train_op_ = self.q_net_opt_.minimize(self.q_net_loss_, var_list = self.q_net_varlist_)
		self.total_loader_ = tf.train.Saver([v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'Adam' not in v.name])
		self.total_saver_ = tf.train.Saver()

	def train_belief_update(self, data_batch, fix_feature):
		# _, cross_entropy, belief_pred, posterior, likelihood, feature = self.sess.run([self.belief_update_train_op_ if fix_feature else self.feature_belief_update_train_op_, \
		# 				self.cross_entropy_, self.belief_pred_, self.belief_pred_1_, self.df_msg_2_norm_, self.feature_extracted_],
		# 				  feed_dict = {self.student_belief_: data_batch['prev_belief'],
		# 				   			   self.message_: data_batch['message'],
		# 				   			   self.distractors_: data_batch['distractors'],
		# 				   			   self.student_belief_spvs_: data_batch['new_belief']})
		
		# return cross_entropy, belief_pred, posterior, likelihood, np.squeeze(feature)
		if fix_feature:
			_, cross_entropy, belief_pred = self.sess.run([self.belief_update_train_op_, \
					self.cross_entropy_, self.belief_pred_],
							feed_dict = {self.student_belief_: data_batch['prev_belief'],
										self.message_: data_batch['message'],
										self.distractors_: data_batch['distractors'],
										self.student_belief_spvs_: data_batch['new_belief']})
			return cross_entropy, belief_pred
		else:
			# _, _, cross_entropy, belief_pred = self.sess.run([self.belief_update_train_op_, self.feature_update_train_op_, \
			# 		self.cross_entropy_, self.belief_pred_],
			# 				feed_dict = {self.student_belief_: data_batch['prev_belief'],
			# 							self.message_: data_batch['message'],
			# 							self.distractors_: data_batch['distractors'],
			# 							self.student_belief_spvs_: data_batch['new_belief']})
			_, cross_entropy, belief_pred, lr = self.sess.run([self.belief_update_train_op_, \
					self.cross_entropy_, self.belief_pred_, self.learning_rate],
							feed_dict = {self.student_belief_: data_batch['prev_belief'],
										self.message_: data_batch['message'],
										self.distractors_: data_batch['distractors'],
										self.student_belief_spvs_: data_batch['new_belief']})
			return cross_entropy, belief_pred, lr

	def pretrain_bayesian_belief_update(self, concept_generator, teacher_pretraining_steps, teacher_pretrain_batch_size,
										teacher_pretrain_ckpt_dir, teacher_pretrain_ckpt_name, continue_steps = 0, silent = False):
		if not os.path.exists(teacher_pretrain_ckpt_dir):
			os.makedirs(teacher_pretrain_ckpt_dir)

		ckpt = tf.train.get_checkpoint_state(teacher_pretrain_ckpt_dir)
		train_steps = teacher_pretraining_steps
		if ckpt:
			self.feature_extract_loader_.restore(self.sess, ckpt.model_checkpoint_path)
			print('Loaded teacher belief update ckpt from %s' % teacher_pretrain_ckpt_dir)
			train_steps = continue_steps
		else:
			print('Cannot loaded teacher belief update ckpt from %s' % teacher_pretrain_ckpt_dir)
		accuracies = []
		l1_diffs = []
		bayesian_wrongs = []
		for ts in range(train_steps):
			data_batch = concept_generator.generate_batch(teacher_pretrain_batch_size)
			cross_entropy, belief_pred, lr = self.train_belief_update(data_batch, fix_feature = False)
			
			l1_diff = np.sum(abs(belief_pred - data_batch['new_belief']), axis = 1)
			correct = (l1_diff <= 5e-2)
			bayesian_wrong = np.mean(np.sum((data_batch['new_belief'] == 0) * (belief_pred > 1e-5), axis = 1) > 0)
			accuracies.append(np.mean(correct))
			l1_diffs.append(np.mean(l1_diff))
			bayesian_wrongs.append(bayesian_wrong)
			if np.sum(np.isnan(belief_pred)) != 0:
				print(belief_pred)
				pdb.set_trace()
			if ts % 100 == 0 and not silent:
				print('[T%d] batch mean cross entropy: %f, mean accuracies: %f, mean l1: %f, bayesian wrong: %f'\
					  % (ts + 1, cross_entropy, np.mean(accuracies), np.mean(l1_diffs), np.mean(bayesian_wrongs)))
				print('new_belief \\ predict_belief: ')
				for n, p in zip(data_batch['new_belief'][:5], belief_pred[:5]):
					print(n)
					print(p)
					print()
				# for n, p, f in zip(data_batch['new_belief'][:5], belief_pred[:5], feature[:5]):
				# 	print(n)
				# 	print(p)
				# 	print(np.transpose(f))
				# 	print()
				# print('prior: ')
				# print(data_batch['prev_belief'][:10])
				# print('likelihood: ')
				# print(likelihood)
				# print('posterior: ')
				# print(posterior)

				boltzman_beta, belief_var_1d = self.sess.run([self.boltzman_beta_, self.belief_var_1d_])
				print('boltzman_beta: %f, belief_var_1d: %f' % (boltzman_beta, belief_var_1d))
				if np.mean(accuracies) > 0.0:
					#idx = np.random.randint(teacher_pretrain_batch_size)
					idx = 3
					for i in range(idx):
						print('\t target:', data_batch['new_belief'][i, :])
						print('\t predict', belief_pred[i, :])
				accuracies = []
				l1_diffs = []
				bayesian_wrongs = []
			if (ts + 1) % 1000 == 0:
				self.feature_belief_saver_.save(self.sess, os.path.join(teacher_pretrain_ckpt_dir,
																   teacher_pretrain_ckpt_name),
											  global_step = teacher_pretraining_steps)
				print('Saved teacher belief update ckpt to %s after %d training'\
				  % (teacher_pretrain_ckpt_dir, ts))
		if train_steps != 0:
			self.feature_belief_saver_.save(self.sess, os.path.join(teacher_pretrain_ckpt_dir,
																   teacher_pretrain_ckpt_name),
											  global_step = teacher_pretraining_steps)
			print('Saved teacher belief update ckpt to %s after %d training'\
				  % (teacher_pretrain_ckpt_dir, train_steps))

	def train_q_net(self, data_batch):
		_, q_net_loss, value = self.sess.run([self.q_net_train_op_, self.q_net_loss_, self.value_],\
									  feed_dict = {self.q_net_spvs_: data_batch['target_q'],
									  			   self.student_belief_: data_batch['student_belief'],
						   			   			   self.message_: data_batch['message'],
						   			   			   self.distractors_: data_batch['distractors'],
						   			   			   self.teacher_belief_: data_batch['teacher_belief']})
		print('Q learning loss: %f' % q_net_loss)
		ridx = np.random.randint(value.shape[0])
		#print(value[ridx], data_batch['target_q'][ridx])
		print('0.8: %f, 0.2: %f' % (np.sum(value * (data_batch['target_q'] == 0.8)) / np.sum(data_batch['target_q'] == 0.8),
									np.sum(value * (data_batch['target_q'] == -0.2)) / np.sum(data_batch['target_q'] == -0.2)))
		print('Teacher value est:', value[ridx: ridx + 10], data_batch['target_q'][ridx: ridx + 10])
		#print(distcts_feat_tensor_3[ridx, :])

		return q_net_loss

	def get_q_value_for_all_msg(self, teacher_belief, student_belief, embeded_concepts):
		
		all_msg_embeddings = np.identity(self.message_space_size_)
		teacher_belief_tile = np.tile(teacher_belief, (self.message_space_size_, 1))
		student_belief_tile = np.tile(student_belief, (self.message_space_size_, 1))
		embeded_concepts_tile = np.tile(embeded_concepts, (self.message_space_size_, 1, 1, 1, 1))

		q_values, belief_pred, distcts_feat_tensor_3, belief_dst, msg_est_tensor = self.sess.run([self.value_, self.belief_pred_, self.distcts_feat_tensor_3_, self.value_, self.msg_est_tensor_2_],
											  	feed_dict = {self.distractors_: embeded_concepts_tile,
														   self.message_: all_msg_embeddings,
														   self.teacher_belief_: teacher_belief_tile,
														   self.student_belief_: student_belief_tile})
		return q_values, belief_pred, distcts_feat_tensor_3, belief_dst, msg_est_tensor[0]

	def update_net(self, belief_update_tuples, q_learning_tuples, update_term = 'Both'):

		debug_structure = {}

		belief_update_batch = {}
		belief_update_batch['prev_belief'] = []
		belief_update_batch['new_belief'] = []
		belief_update_batch['message'] = []
		belief_update_batch['distractors'] = []
		for belief_tuple in belief_update_tuples:
			belief_update_batch['distractors'].append(belief_tuple[0])
			belief_update_batch['prev_belief'].append(belief_tuple[1])
			belief_update_batch['message'].append(belief_tuple[2])
			belief_update_batch['new_belief'].append(belief_tuple[3])

		for k in belief_update_batch:
			belief_update_batch[k] = np.array(belief_update_batch[k])
		if update_term == 'Both' or update_term == 'Belief':
			cross_entropy, belief_pred = self.train_belief_update(belief_update_batch, fix_feature = True)
			print('Teacher\'s belief esimate cross_entropy: %f' % cross_entropy)

			debug_structure['teacher_belief_prediction'] = belief_pred

		q_learning_batch = {}
		q_learning_batch['student_belief'] = []
		q_learning_batch['teacher_belief'] = []
		q_learning_batch['message'] = []
		q_learning_batch['distractors'] = []
		q_learning_batch['target_q'] = []
		for q_learning_tuple in q_learning_tuples:
			q_learning_batch['distractors'].append(q_learning_tuple[0])
			q_learning_batch['student_belief'].append(q_learning_tuple[1])
			q_learning_batch['teacher_belief'].append(q_learning_tuple[2])
			q_learning_batch['message'].append(q_learning_tuple[3])
			q_learning_batch['target_q'].append(q_learning_tuple[4])


		for k in q_learning_batch:
			q_learning_batch[k] = np.array(q_learning_batch[k])
		if update_term == 'Both' or update_term == 'Q-Net':
			q_net_loss = self.train_q_net(q_learning_batch)

		return debug_structure
		

if __name__ == '__main__':
	main()
