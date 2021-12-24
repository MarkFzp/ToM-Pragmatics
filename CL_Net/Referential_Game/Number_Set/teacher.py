import os
import math
import numpy as np
import tensorflow as tf

from concept import Concept
import pdb
np.set_printoptions(precision=5, suppress=True)

class Teacher:
	def __init__(self, sess, rl_gamma, boltzman_beta,
				 belief_var_1d, num_distractors, attributes_size,
				 message_space_size):
		self.sess = sess
		self.num_distractors_ = num_distractors
		self.attributes_size_ = attributes_size
		self.message_space_size_ = message_space_size
		self.rl_gamma_ = rl_gamma
		self.boltzman_beta_ = boltzman_beta
		self.belief_var_1d_ = belief_var_1d
		################
		# Placeholders #
		################
		with tf.variable_scope('Teacher'):
			self.distractors_ = tf.placeholder(tf.float32, name = 'distractors', 
											  shape = [None, self.num_distractors_, self.attributes_size_])
			self.distractors_tensor_ = tf.expand_dims(self.distractors_, 2)
			self.message_ = tf.placeholder(tf.float32, shape = [None, self.message_space_size_], name = 'message')

			self.teacher_belief_ = tf.placeholder(tf.float32, shape = [None, self.num_distractors_], name = 'teacher_belief')
			self.student_belief_ = tf.placeholder(tf.float32, shape = [None, self.num_distractors_], name = 'student_belief')
			self.student_belief_spvs_ = tf.placeholder(tf.float32, shape = [None, self.num_distractors_], name = 'student_belief_spvs')
			
			self.q_net_spvs_ = tf.placeholder(tf.float32, shape = [None])
		########################
		# Belief Update Module #
		########################
		self.belief_update_opt_ = tf.train.AdamOptimizer(learning_rate = 1e-3)
		with tf.variable_scope('Belief_Update'):
			self.df1_ = tf.layers.conv2d(self.distractors_tensor_, 3 * self.message_space_size_, kernel_size = [1, 1],
								   		kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1),
								   		activation = tf.nn.leaky_relu)
			self.df2_ = tf.layers.conv2d(self.df1_, 3 * self.message_space_size_, kernel_size = [1, 1],
								   		kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-1),
								   		activation = tf.nn.leaky_relu)
			# self.df3_ = tf.layers.conv2d(self.df2_, 1 * self.message_space_size_, kernel_size = [1, 1],
			# 					   		kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2),
			# 					   		activation = None)

			self.msg_from_df_1_ = []
			for _ in range(self.num_distractors_):
				self.msg_from_df_1_.append(tf.layers.conv2d(self.df2_, 2 * self.message_space_size_, kernel_size = [self.num_distractors_, 1],
														  kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-1),
														  padding = 'valid', activation = tf.nn.leaky_relu))

			self.msg_est_tensor_1_ = tf.concat(self.msg_from_df_1_, axis = 1)

			self.msg_from_df_2_ = []
			for _ in range(self.num_distractors_):
				self.msg_from_df_2_.append(tf.layers.conv2d(self.msg_est_tensor_1_, 1 * self.message_space_size_, kernel_size = [self.num_distractors_, 1],
														  kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2),
														  padding = 'valid', activation = None))

			self.msg_est_tensor_2_ = tf.concat(self.msg_from_df_2_, axis = 1)
			
			self.reg_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('Belief')]
			#######################
			#network belief update#
			#######################
			self.msg_est_tensor_2d_ = tf.squeeze(self.msg_est_tensor_2_, axis = 2)
			
			self.belief_var_1d_ = tf.exp(tf.Variable(initial_value = self.belief_var_1d_, trainable = True, dtype = tf.float32))
			# self.belief_var_ = tf.layers.conv2d(self.msg_est_tensor_3_, 1, kernel_size = [self.num_distractors_, 1],
			# 									kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-3),
			# 									padding = 'valid', activation = None)
			# self.belief_var_1d_ = tf.squeeze(self.belief_var_, axis = 2)
			self.boltzman_beta_ = tf.Variable(initial_value = self.boltzman_beta_, trainable = False, dtype = tf.float32, name = 'boltzman_beta')

			
			self.msg_indices_ = tf.where(tf.not_equal(self.message_, 0))
			
			self.df_msg_match_ = tf.exp(self.boltzman_beta_ * self.msg_est_tensor_2d_)
			self.df_msg_match_norm_ =  tf.div_no_nan(self.df_msg_match_, tf.reduce_sum(self.df_msg_match_, axis = 2, keepdims = True))
			
			self.df_msg_2_norm_ = tf.gather_nd(tf.transpose(self.df_msg_match_norm_, perm = [0, 2, 1]),
													  self.msg_indices_)
						
			#self.df_msg_1_ = tf.multiply(self.dfb_merge_pre_3_, tf.expand_dims(tf.expand_dims(self.message_, 1), 1))
			#self.df_msg_2_ = tf.exp(self.boltzman_beta_ * tf.reduce_sum(tf.squeeze(self.df_msg_1_, 2), axis = 2))
			#self.df_msg_2_norm_ = tf.nn.relu(self.df_msg_2_ + self.belief_var_1_)

			self.belief_pred_1_ = tf.multiply(self.df_msg_2_norm_, self.student_belief_)
			self.belief_pred_full_ = tf.concat([self.belief_pred_1_, self.belief_var_1d_ * tf.slice(tf.ones_like(self.belief_pred_1_), [0, 0], [-1, 1])], axis = 1)
			
			#######################
			#network belief update#
			#######################
			
			'''
			######################
			#kernel belief update#
			######################
			self.kernel_columns_ = []
			for i in range(self.num_distractors_):
				self.df_msg_1_ = tf.multiply(self.msg_est_tensor_2_, tf.expand_dims(tf.expand_dims(self.message_, 1), 1))
				self.df_msg_2_ = tf.contrib.layers.fully_connected(tf.layers.flatten(self.df_msg_1_),\
																  2 * self.num_distractors_, activation_fn = tf.nn.leaky_relu)
				self.df_msg_3_ = tf.contrib.layers.fully_connected(self.df_msg_2_,
																  self.num_distractors_, activation_fn = None)
				kernel_column = tf.nn.relu(self.df_msg_3_)
				self.kernel_columns_.append(tf.expand_dims(tf.div_no_nan(kernel_column,
														  tf.reduce_sum(kernel_column, axis = 1, keepdims = True)), -1))
			self.kernel_pre_norm_ = tf.no_op()
			self.kernel_ = tf.concat(self.kernel_columns_, axis = 2)
			print('<Belief Update Kernel Generator Constructed>')
			self.belief_pred_ = tf.nn.relu(tf.squeeze(tf.matmul(self.kernel_, tf.expand_dims(self.student_belief_, -1)), -1))
			######################
			#kernel belief update#
			######################
			'''
			self.belief_pred_full_norm_ = tf.div_no_nan(self.belief_pred_full_, tf.reduce_sum(self.belief_pred_full_, axis = 1, keepdims = True))
			self.belief_pred_ = tf.slice(self.belief_pred_full_norm_, [0, 0], [-1, self.num_distractors_])
			self.regularization_ = 1e-4 * tf.add_n([ tf.nn.l2_loss(v) for v in self.reg_varlist_ if 'bias' not in v.name ])
			self.cross_entropy_1_ = -1 * tf.reduce_mean(tf.reduce_sum(tf.multiply(self.student_belief_spvs_, tf.math.log(self.belief_pred_)), axis = 1))
			self.cross_entropy_2_ = -1 * tf.reduce_mean(tf.reduce_sum(tf.multiply(self.belief_pred_, tf.math.log(self.student_belief_spvs_ + 1e-9)), axis = 1))
			self.cross_entropy_ = self.cross_entropy_1_ + self.cross_entropy_2_ + self.regularization_
		self.belief_train_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('Belief_Update')]
		self.belief_update_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name.startswith('Belief_Update')]
		self.belief_update_train_op_ = self.belief_update_opt_.minimize(self.cross_entropy_, var_list = self.belief_train_varlist_)
		self.belief_update_saver_ = tf.train.Saver()
		self.belief_update_loader_ = tf.train.Saver(self.belief_update_varlist_)
		####################
		# Q-network Module #
		####################
		self.q_net_opt_ = tf.train.AdamOptimizer(learning_rate = 1e-5)
		with tf.variable_scope('q_net'):
			self.distct_feat_1_ = tf.layers.conv2d(self.distractors_tensor_, 3 * self.message_space_size_, kernel_size = [1, 1],
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
														  padding = 'valid', activation = tf.nn.leaky_relu))

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
			'''

			
			self.df_b1_ = tf.multiply(tf.squeeze(self.distct_feat_2_, axis = 2), tf.expand_dims(self.teacher_belief_, -1))
			self.df_b2_ = tf.multiply(tf.squeeze(self.distct_feat_2_, axis = 2), tf.expand_dims(self.belief_pred_, -1))
			self.concat_df_b_ = tf.layers.flatten(tf.concat((self.df_b1_, self.df_b2_), axis = 2))
			# self.dfb_merge_pre_ = tf.contrib.layers.fully_connected(tf.reduce_sum(tf.abs(self.df_b1_ - self.df_b2_), axis = 1), 4, activation_fn = tf.nn.leaky_relu,
			# 													weights_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2))
			self.dfb_merge_pre_1_ = tf.contrib.layers.fully_connected(self.concat_df_b_, 6, activation_fn = tf.nn.leaky_relu,
																weights_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2))
			self.dfb_merge_pre_2_ = tf.contrib.layers.fully_connected(self.dfb_merge_pre_1_, 4, activation_fn = tf.nn.leaky_relu,
																weights_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2))
			self.dfb_merge_ = tf.contrib.layers.fully_connected(self.dfb_merge_pre_2_, 1, activation_fn = None,
																weights_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2))
			self.value_ = tf.squeeze(self.dfb_merge_)
			'''
			# self.dfb_merge_ = tf.reduce_sum(tf.square(self.df_b1_ - self.df_b2_), axis = [1, 2])
			
			# self.value_param_0_ = tf.squeeze(tf.contrib.layers.fully_connected(self.concat_df_b_, 1, activation_fn = None))
			# self.value_param_00_ = tf.squeeze(tf.contrib.layers.fully_connected(self.dfb_merge_pre_1_, 1, activation_fn = None))
			# self.value_param_000_ = tf.squeeze(tf.contrib.layers.fully_connected(self.dfb_merge_pre_1_, 1, activation_fn = None))
			# self.value_param_0000_ = tf.squeeze(tf.contrib.layers.fully_connected(self.dfb_merge_pre_1_, 1, activation_fn = None))
			# self.value_param_1_ = tf.Variable(initial_value = -1, trainable = True, dtype = tf.float32)
			# self.value_param_2_ = tf.Variable(initial_value = 1, trainable = True, dtype = tf.float32)
			# self.value_param_3_ = tf.Variable(initial_value = -1, trainable = True, dtype = tf.float32)
			
			#self.value_param_2_ * tf.exp(self.value_param_1_ * tf.squeeze(self.dfb_merge_)) + self.value_param_3_
			#self.value_ = 1 - tf.squeeze(tf.contrib.layers.fully_connected(tf.reduce_sum(self.df_b1_ - self.df_b2_, axis = 2), 1, activation_fn = None))
			self.reg_varlist_q_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('q_net')]
			self.regularization_q_ = 1e-4 * tf.add_n([ tf.nn.l2_loss(v) for v in self.reg_varlist_q_ if 'bias' not in v.name ])
			self.q_net_loss_pre_ = tf.square(self.value_ - self.q_net_spvs_)
			self.success_mask_ = tf.to_float(tf.math.greater(self.q_net_spvs_, 0.0))
			self.fail_mask_ = tf.to_float(tf.math.greater(0.0, self.q_net_spvs_))
			self.imbalance_penalty_ = self.success_mask_ + self.fail_mask_ * tf.div_no_nan(tf.reduce_sum(self.success_mask_), tf.reduce_sum(self.fail_mask_))
			#self.q_net_loss_ = tf.reduce_mean(self.q_net_loss_pre_ * tf.to_float(self.q_net_loss_pre_ > 0.05) * self.imbalance_penalty_) + self.regularization_q_
			self.q_net_loss_ = tf.reduce_mean(self.q_net_loss_pre_ * self.imbalance_penalty_) + self.regularization_q_
			self.q_net_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('q_net')]
			self.q_net_train_op_ = self.q_net_opt_.minimize(self.q_net_loss_, var_list = self.q_net_varlist_)
		self.total_loader_ = tf.train.Saver([v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'Adam' not in v.name])
		self.total_saver_ = tf.train.Saver()

	def train_belief_update(self, data_batch):
		_, cross_entropy, belief_pred, posterior, likelihood = self.sess.run([self.belief_update_train_op_, self.cross_entropy_, self.belief_pred_, self.belief_pred_1_, self.df_msg_2_norm_],
						  feed_dict = {self.student_belief_: data_batch['prev_belief'],
						   			   self.message_: data_batch['message'],
						   			   self.distractors_: data_batch['distractors'],
						   			   self.student_belief_spvs_: data_batch['new_belief']})
		
		return cross_entropy, belief_pred, posterior[:10], likelihood[:10]

	def pretrain_bayesian_belief_update(self, concept_generator, teacher_pretraining_steps, teacher_pretrain_batch_size,
										teacher_pretrain_ckpt_dir, teacher_pretrain_ckpt_name, continue_steps = 0, silent = False):
		if not os.path.exists(teacher_pretrain_ckpt_dir):
			os.makedirs(teacher_pretrain_ckpt_dir)

		ckpt = tf.train.get_checkpoint_state(teacher_pretrain_ckpt_dir)
		train_steps = teacher_pretraining_steps
		if ckpt:
			self.belief_update_loader_.restore(self.sess, ckpt.model_checkpoint_path)
			print('Loaded teacher belief update ckpt from %s' % teacher_pretrain_ckpt_dir)
			train_steps = continue_steps
		else:
			print('Cannot loaded teacher belief update ckpt from %s' % teacher_pretrain_ckpt_dir)
			
		accuracies = []
		l1_diffs = []
		bayesian_wrongs = []
		for ts in range(train_steps):
			data_batch = concept_generator.generate_batch(teacher_pretrain_batch_size)
			cross_entropy, belief_pred, posterior, likelihood = self.train_belief_update(data_batch)
			
			l1_diff = np.sum(abs(belief_pred - data_batch['new_belief']), axis = 1)
			correct = (l1_diff <= 5e-2)
			bayesian_wrong = np.mean(np.sum((data_batch['new_belief'] == 0) * (belief_pred > 1e-5), axis = 1) > 0)
			accuracies.append(np.mean(correct))
			l1_diffs.append(np.mean(l1_diff))
			bayesian_wrongs.append(bayesian_wrong)
			if np.sum(np.isnan(belief_pred)) != 0:
				pdb.set_trace()
			if ts % 1000 == 0 and not silent:
				print('[T%d] batch mean cross entropy: %f, mean accuracies: %f, mean l1: %f, bayesian wrong: %f'\
					  % (ts + 1, cross_entropy, np.mean(accuracies), np.mean(l1_diffs), np.mean(bayesian_wrongs)))
				boltzman_beta, belief_var_1d = self.sess.run([self.boltzman_beta_, self.belief_var_1d_])
				print('boltzman_beta: %f, belief_var_1d: %f' % (boltzman_beta, belief_var_1d))
				print('new_belief: ')
				print(data_batch['new_belief'][:10])
				print('prior: ')
				print(data_batch['prev_belief'][:10])
				print('likelihood: ')
				print(likelihood)
				print('posterior: ')
				print(posterior)
				print('predict_belief: ')
				print(belief_pred[:10])
				if np.mean(accuracies) > 0.9:
					#idx = np.random.randint(teacher_pretrain_batch_size)
					idx = teacher_pretrain_batch_size
					for i in range(idx):
						print('\t target:', data_batch['new_belief'][i, :])
						print('\t predict', belief_pred[i, :])
				accuracies = []
				l1_diffs = []
				bayesian_wrongs = []
			if (ts + 1) % 10000 == 0:
				self.belief_update_saver_.save(self.sess, os.path.join(teacher_pretrain_ckpt_dir,
																   teacher_pretrain_ckpt_name),
											  global_step = teacher_pretraining_steps)
				print('Saved teacher belief update ckpt to %s after %d training'\
				  % (teacher_pretrain_ckpt_dir, ts))
		if train_steps != 0:
			self.belief_update_saver_.save(self.sess, os.path.join(teacher_pretrain_ckpt_dir,
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
		embeded_concepts_tile = np.tile(embeded_concepts, (self.message_space_size_, 1, 1))

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
			cross_entropy, belief_pred = self.train_belief_update(belief_update_batch)
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
