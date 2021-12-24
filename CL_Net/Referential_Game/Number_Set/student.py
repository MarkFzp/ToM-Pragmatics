import os

import tensorflow as tf
import numpy as np

import pdb

class Student:
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
		with tf.variable_scope('Student'):
			self.distractors_ = tf.placeholder(tf.float32, name = 'distractors', 
											   shape = [None, self.num_distractors_, self.attributes_size_])
			self.distractors_tensor_ = tf.expand_dims(self.distractors_, 2)
			self.message_ = tf.placeholder(tf.float32, shape = [None, self.message_space_size_], name = 'message')
			self.student_belief_ = tf.placeholder(tf.float32, shape = [None, self.num_distractors_], name = 'student_belief')
			self.advantage_ = tf.placeholder(tf.float32, shape = [None], name = 'advantage')
			self.prediction_indices_ = tf.placeholder(tf.int32, shape = [None, 2])
			self.pretrain_belief_spvs_ = tf.placeholder(tf.float32, shape = [None, self.num_distractors_], name = 'pretrain_student_belief')
			self.bayesian_belief_spvs_ = tf.placeholder(tf.float32, shape = [None, self.num_distractors_], name = 'bayesian_belief')
			
			#actor-critic: critic
			self.teacher_belief_ = tf.placeholder(tf.float32, shape = [None, self.num_distractors_], name = 'correct_answer')
			self.critic_value_spvs_ = tf.placeholder(tf.float32, shape = [None])
		####################
		# Q-Net Estimation #
		####################
		with tf.variable_scope('Belief_Network'):
			self.df1_ = tf.layers.conv2d(self.distractors_tensor_, 3 * self.message_space_size_, kernel_size = [1, 1],
								   		 kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1),
								   		 activation = tf.nn.leaky_relu)
			self.df2_ = tf.layers.conv2d(self.df1_, 3 * self.message_space_size_, kernel_size = [1, 1],
								   		 kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-1),
								   		 activation = tf.nn.leaky_relu)
			# #teacher belief is one-hot, thus can be used as after applying teacher belief mask
			# self.df3_ = tf.layers.conv2d(self.df2_, 1 * self.message_space_size_, kernel_size = [1, 1],
			# 					   		 kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2),
			# 					   		 activation = None)
			# self.df3_2d_ = tf.squeeze(self.df3_, axis = 2)

			self.msg_from_df_1_ = []
			for _ in range(self.num_distractors_):
				self.msg_from_df_1_.append(tf.layers.conv2d(self.df2_, 2 * self.message_space_size_, kernel_size = [self.num_distractors_, 1],
														  kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-1),
														  padding = 'valid', activation = tf.nn.leaky_relu))

			# self.dist_feat_var_ = tf.Variable(initial_value = 0.5, trainable = True, dtype = tf.float32)
			self.msg_est_tensor_1_ = tf.concat(self.msg_from_df_1_, axis = 1)

			self.msg_from_df_2_ = []
			for _ in range(self.num_distractors_):
				self.msg_from_df_2_.append(tf.layers.conv2d(self.msg_est_tensor_1_, self.message_space_size_, kernel_size = [self.num_distractors_, 1],
														  kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2),
														  padding = 'valid', activation = None))

			self.msg_est_tensor_2_ = tf.concat(self.msg_from_df_2_, axis = 1)


			# self.dist_feat_var_ = tf.Variable(initial_value = 0.5, trainable = True, dtype = tf.float32)
			
			self.reg_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('Belief')]

			#######################
			#network belief update#
			#######################
			self.msg_est_tensor_2d_ = tf.squeeze(self.msg_est_tensor_2_, axis = 2)
			#self.msg_est_tensor_median_ = tf.contrib.distributions.percentile(self.msg_est_tensor_2d_, 0.5, axis = 1, keep_dims = True)
			#self.msg_est_tensor_median_ = tf.reduce_mean(self.msg_est_tensor_2d_, axis = 1, keepdims = True)
			#self.msg_est_tensor_2d_ -= self.msg_est_tensor_median_

			self.msg_indices_ = tf.where(tf.not_equal(self.message_, 0))
			#self.msg_mask_ = tf.expand_dims(tf.expand_dims(tf.constant(np.identity(self.message_space_size_), dtype = tf.float32), 0), 0)

			self.belief_var_1d_ = tf.exp(tf.Variable(initial_value = self.belief_var_1d_, trainable = True, dtype = tf.float32))
			# self.belief_var_ = tf.layers.conv2d(self.msg_est_tensor_3_, 1, kernel_size = [self.num_distractors_, 1],
			# 									kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-3),
			# 									padding = 'valid', activation = None)
			# self.belief_var_1d_ = tf.squeeze(self.belief_var_, axis = 2)
			self.boltzman_beta_ = tf.Variable(initial_value = self.boltzman_beta_, trainable = False, dtype = tf.float32, name = 'boltzman_beta')
			self.df_msg_match_ = tf.exp(self.boltzman_beta_ * self.msg_est_tensor_2d_)
			self.df_msg_match_norm_ =  tf.div_no_nan(self.df_msg_match_, tf.reduce_sum(self.df_msg_match_, axis = 2, keepdims = True))

			self.df_msg_compatibility_ = tf.gather_nd(tf.transpose(self.df_msg_match_norm_, perm = [0, 2, 1]),
														self.msg_indices_)

			self.obverter_loss_ = -1 * tf.reduce_sum(self.teacher_belief_ * self.df_msg_compatibility_)


			self.likelihood_ = tf.identity(self.df_msg_compatibility_)

			self.new_student_belief_1_ = tf.multiply(self.likelihood_, self.student_belief_)
			self.new_student_belief_full_ = tf.concat([self.new_student_belief_1_, self.belief_var_1d_ * tf.slice(tf.ones_like(self.new_student_belief_1_), [0, 0], [-1, 1])], axis = 1)
			#######################
			#network belief update#
			#######################

			'''
			######################
			#kernel belief update#
			#######################
			self.kernel_columns_ = []
			self.df_msg_compatibility_ = tf.no_op()
			for i in range(self.num_distractors_):
				self.df_msg_1_ = tf.multiply(self.msg_est_tensor_2_, tf.expand_dims(tf.expand_dims(self.message_, 1), 1))
				self.df_msg_2_ = tf.contrib.layers.fully_connected(tf.layers.flatten(self.df_msg_1_),\
																  2 * self.num_distractors_, activation_fn = tf.nn.leaky_relu)
				self.df_msg_3_ = tf.contrib.layers.fully_connected(self.df_msg_2_,
																  self.num_distractors_, activation_fn = None)
				kernel_column = tf.math.exp(self.df_msg_3_)
				#kernel_column = tf.math.exp(tf.reduce_sum(self.df_msg_1, axis = [2, 3]))
				self.kernel_columns_.append(tf.expand_dims(tf.div(kernel_column,
														  tf.reduce_sum(kernel_column, axis = 1, keepdims = True)), -1))
			self.kernel_ = tf.concat(self.kernel_columns_, axis = 2)
			print('<Belief Update Kernel Generator Constructed>')
			self.new_student_belief_ = tf.squeeze(tf.matmul(self.kernel_, tf.expand_dims(self.student_belief_, -1)), -1)
			######################
			#kernel belief update#
			######################
			'''
			self.new_student_belief_full_norm_ = tf.div_no_nan(self.new_student_belief_full_,
												   		  tf.reduce_sum(self.new_student_belief_full_,
												   				 		axis = 1, keepdims = True))
			self.new_student_belief_norm_ = tf.slice(self.new_student_belief_full_norm_, [0, 0], [-1, self.num_distractors_])
			self.new_student_belief_norm_sum_ = tf.reduce_sum(self.new_student_belief_norm_, axis = 1, keepdims = True)

		with tf.variable_scope('Value_Network'):
			# self.distct_feat_1_ = tf.layers.conv2d(self.distractors_tensor_, 3 * self.message_space_size_, kernel_size = [1, 1],
			# 					   				   kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2),
			# 					   				   activation = tf.nn.leaky_relu)
			# self.distct_feat_2_ = tf.layers.conv2d(self.distct_feat_1_, 2 * self.message_space_size_, kernel_size = [1, 1],
			# 					   				   kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-1),
			# 					   				   activation = tf.nn.leaky_relu)
			self.distct_feat_2_weighted_ = tf.multiply(self.distractors_tensor_, tf.expand_dims(tf.expand_dims(self.student_belief_, -1), -1))

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
			self.value_param_1_ = tf.Variable(initial_value = -0.2, trainable = False, dtype = tf.float32)
			self.critic_value_ = tf.reduce_sum(tf.multiply(tf.squeeze(self.distcts_feat_tensor_3_), self.teacher_belief_), axis = 1) + self.value_param_1_
			# self.df_b2_ = tf.multiply(tf.squeeze(self.distcts_feat_tensor_2_, axis = 2), tf.expand_dims(self.student_belief_, -1))
			# self.concat_df_b_ = tf.layers.flatten(tf.concat((self.df_b1_, self.df_b2_), axis = 2))

			# self.dfb_merge_pre_1_ = tf.contrib.layers.fully_connected(self.concat_df_b_, 6, activation_fn = tf.nn.leaky_relu,
			# 													weights_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2))
			# self.dfb_merge_pre_2_ = tf.contrib.layers.fully_connected(self.dfb_merge_pre_1_, 4, activation_fn = tf.nn.leaky_relu,
			# 													weights_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2))
			# self.dfb_merge_ = tf.contrib.layers.fully_connected(self.dfb_merge_pre_2_, 1, activation_fn = tf.nn.tanh,
			# 													weights_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2))
			# self.critic_value_ = tf.squeeze(self.dfb_merge_)
			'''
			self.critic_value_param_1_ = tf.Variable(initial_value = -1, trainable = True, dtype = tf.float32)
			self.critic_value_param_2_ = tf.Variable(initial_value = 0, trainable = True, dtype = tf.float32)
			self.critic_value_param_3_ = tf.Variable(initial_value = -1, trainable = True, dtype = tf.float32)
			self.critic_value_ = tf.exp(self.critic_value_param_1_ * (tf.squeeze(self.dfb_merge_) +\
										self.critic_value_param_2_)) +\
								 self.critic_value_param_3_
			'''
	

		with tf.variable_scope('Policy_Network'):	
			# self.var_process_factor_0_ = tf.contrib.layers.fully_connected(self.new_student_belief_norm_, 4, activation_fn = tf.nn.leaky_relu)
			# self.var_process_factor_00_ = tf.contrib.layers.fully_connected(self.var_process_factor_0_, 4, activation_fn = tf.nn.leaky_relu)
			# self.var_process_factor_000_ = tf.contrib.layers.fully_connected(self.var_process_factor_00_, 4, activation_fn = tf.nn.leaky_relu)

			
			#use entropy to delay final prediction
			self.new_belief_var_ = tf.reduce_sum(-1 * self.new_student_belief_full_norm_ * tf.math.log(self.new_student_belief_full_norm_ + 1e-9), axis = 1, keepdims = True)
			self.var_process_factor_1_ = tf.Variable(initial_value = 1, trainable = True, dtype = tf.float32)
			self.var_process_factor_2_ = tf.Variable(initial_value = 0, trainable = True, dtype = tf.float32)
			self.var_process_factor_3_ = tf.Variable(initial_value = 0, trainable = True, dtype = tf.float32)
			#self.new_belief_var_process_ = tf.exp(self.var_process_factor_00_) / (tf.pow(self.new_belief_var_, self.var_process_factor_000_)) + self.var_process_factor_0_
			self.new_belief_var_process_ = tf.exp(self.var_process_factor_1_ * self.new_belief_var_) +\
												  self.var_process_factor_2_ * self.new_belief_var_ + self.var_process_factor_3_
			self.pre_prediction_ = tf.concat([self.new_student_belief_full_norm_, tf.nn.relu(self.new_belief_var_process_)], axis = 1)
			
			self.student_prediction_ = tf.div_no_nan(self.pre_prediction_, tf.reduce_sum(self.pre_prediction_, axis = 1, keepdims = True))
			
		## pretrain varaibles ##
		self.regularization_ = 1e-4 * tf.add_n([ tf.nn.l2_loss(v) for v in self.reg_varlist_ if 'bias' not in v.name ])
		self.cross_entropy_1_ = -1 * tf.reduce_mean(tf.reduce_sum(tf.multiply(self.pretrain_belief_spvs_, tf.math.log(self.new_student_belief_norm_)), axis = 1))
		self.cross_entropy_2_ = -1 * tf.reduce_mean(tf.reduce_sum(tf.multiply(self.new_student_belief_norm_, tf.math.log(self.pretrain_belief_spvs_ + 1e-9)), axis = 1))
		self.cross_entropy_ = self.cross_entropy_1_ + self.cross_entropy_2_ + self.regularization_
		self.pretrain_opt_ = tf.train.AdamOptimizer(learning_rate = 1e-5)
		self.pretrain_train_op_ = self.pretrain_opt_.minimize(self.cross_entropy_)
		self.pretrain_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)\
								  if v.name.startswith('Belief_Network')]								  
		self.pretrain_saver_ = tf.train.Saver()
		self.pretrain_loader_ = tf.train.Saver([v for v in self.pretrain_varlist_ if 'Adam' not in v.name])

		## policy gradient ##
		self.pg_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)\
							if v.name.startswith('Policy_Network')]
		
		self.bn_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)\
							if v.name.startswith('Belief_Network')]

		self.new_belief_after_mask_ = tf.reduce_sum(tf.cast(tf.equal(self.bayesian_belief_spvs_, 0.0), tf.float32) *\
									  tf.cast(tf.greater_equal(self.new_student_belief_norm_, 1e-4), tf.float32) *\
									  self.new_student_belief_norm_, axis = 1)
		self.bayesian_violate_loss_ = -1 * tf.reduce_mean(tf.log(tf.where(tf.equal(self.new_belief_after_mask_, 0),
																		  tf.ones_like(self.new_belief_after_mask_), 
																		  self.new_belief_after_mask_)))

		self.act_prob_ = tf.math.log(tf.gather_nd(self.student_prediction_, self.prediction_indices_))
		self.loss_pg_ = -1 * tf.reduce_sum(tf.multiply(self.act_prob_, self.advantage_)) + 1e-1 * self.regularization_

		self.policy_gradient_opt_ = tf.train.AdamOptimizer(learning_rate = 1e-4)
		self.policy_gradient_train_op_1_ = self.policy_gradient_opt_.minimize(self.loss_pg_, var_list = self.pg_varlist_)
		self.policy_gradient_opt_finer_ = tf.train.AdamOptimizer(learning_rate = 1e-3)
		self.policy_gradient_train_op_2_ = tf.no_op()#self.policy_gradient_opt_finer_.minimize(self.obverter_loss_, var_list = self.bn_varlist_)
		self.policy_gradient_train_op_3_ = self.policy_gradient_opt_finer_.minimize(self.loss_pg_, var_list = self.bn_varlist_)

		## value iteration ##
		self.vi_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)\
							if v.name.startswith('Value_Network')]
		self.success_mask_ = tf.to_float(tf.math.greater(self.critic_value_spvs_, 0.0))
		self.fail_mask_ = tf.to_float(tf.math.greater(0.0, self.critic_value_spvs_))
		self.imbalance_penalty_ = self.success_mask_ + self.fail_mask_ * 1
		self.loss_vi_pre_ = tf.square(self.critic_value_ - self.critic_value_spvs_)
		self.loss_vi_ = tf.reduce_mean(self.loss_vi_pre_ * tf.to_float(self.loss_vi_pre_ > 0.05) * self.imbalance_penalty_)

		self.value_iteration_opt_ = tf.train.AdamOptimizer(learning_rate = 1e-4)
		self.value_iteration_train_op_ = self.value_iteration_opt_.minimize(self.loss_vi_, var_list = self.vi_varlist_)
		self.total_saver_ = tf.train.Saver()
		self.total_loader_ = tf.train.Saver([v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'Adam' not in v.name])
		self.value_saver_ = tf.train.Saver(list(set(self.vi_varlist_)))

	
	def pretrain_belief_update(self, data_batch):
		_, cross_entropy, belief_pred, df_msg_compatibility = self.sess.run([self.pretrain_train_op_,
													   self.cross_entropy_,
													   self.new_student_belief_norm_,
													   self.df_msg_compatibility_],
													  feed_dict = {self.student_belief_: data_batch['prev_belief'],
													   			   self.message_: data_batch['message'],
													   			   self.distractors_: data_batch['distractors'],
													   			   self.pretrain_belief_spvs_: data_batch['new_belief']})
		return cross_entropy, belief_pred, df_msg_compatibility

	def pretrain_bayesian_belief_update(self, concept_generator, student_pretraining_steps, student_pretrain_batch_size,
										student_pretrain_ckpt_dir, student_pretrain_ckpt_name, continue_steps = 0, silent = False):
		if not os.path.exists(student_pretrain_ckpt_dir):
			os.makedirs(student_pretrain_ckpt_dir)

		ckpt = tf.train.get_checkpoint_state(student_pretrain_ckpt_dir)
		train_steps = student_pretraining_steps
		if ckpt:
			self.pretrain_loader_.restore(self.sess, ckpt.model_checkpoint_path)
			print('Loaded student belief update ckpt from %s' % student_pretrain_ckpt_dir)
			train_steps = continue_steps
		else:
			print('Cannot loaded student belief update ckpt from %s' % student_pretrain_ckpt_dir)
			
		accuracies = []
		l1_diffs = []
		bayesian_wrongs = []
		for ts in range(train_steps):
			data_batch = concept_generator.generate_batch(student_pretrain_batch_size)
			cross_entropy, belief_pred, df_msg_compatibility = self.pretrain_belief_update(data_batch)			
			l1_diff = np.sum(abs(belief_pred - data_batch['new_belief']), axis = 1)
			correct = (l1_diff <= 5e-2)
			bayesian_wrong = np.mean(np.sum((data_batch['new_belief'] == 0) * (belief_pred > 1e-5), axis = 1) > 0)
			accuracies.append(np.mean(correct))
			l1_diffs.append(np.mean(l1_diff))
			bayesian_wrongs.append(bayesian_wrong)
			if ts % 1000 == 0 and not silent:
				print('[S%d] batch mean cross entropy: %f, mean accuracies: %f, mean l1: %f, bayesian wrong: %f'\
					  % (ts + 1, cross_entropy, np.mean(accuracies), np.mean(l1_diffs), np.mean(bayesian_wrongs)))
				boltzman_beta, belief_var_1d = self.sess.run([self.boltzman_beta_, self.belief_var_1d_])
				print('boltzman_beta: %f, belief_var_1d: %f' % (boltzman_beta, belief_var_1d))
				if np.mean(accuracies) >= 0.9:
					idx = np.random.randint(student_pretrain_batch_size)
					print('\t target:', data_batch['new_belief'][idx, :])
					print('\t predict', belief_pred[idx, :])
				accuracies = []
				l1_diffs = []
				bayesian_wrongs = []
			if (ts + 1) % 10000 == 0:
				self.pretrain_saver_.save(self.sess, os.path.join(student_pretrain_ckpt_dir,
																   student_pretrain_ckpt_name),
											  global_step = student_pretraining_steps)
				print('Saved student belief update ckpt to %s after %d training'\
					  % (student_pretrain_ckpt_dir, ts))
		if train_steps != 0:
			self.pretrain_saver_.save(self.sess, os.path.join(student_pretrain_ckpt_dir,
																   student_pretrain_ckpt_name),
											  global_step = student_pretraining_steps)
			print('Saved student belief update ckpt to %s after %d training'\
				  % (student_pretrain_ckpt_dir, train_steps))

	# trajectory_batch = [[(distractors, belief, msg, action, advantage), ...], [], []]
	# advantage should include 1/m, where m is the batch size
	def update_net(self, trajectory_batch, phase = 1):

		#self.sess.run([self.reset_gradients_])
		
		num_traject = len(trajectory_batch)
		max_length = max([len(traject) for traject in trajectory_batch])
		step_idx = 0
		data_batch = {}

		data_batch['distractors'] = []
		data_batch['belief'] = []
		data_batch['message'] = []
		data_batch['action'] = []
		data_batch['advantage'] = []
		data_batch['critic_value'] = []
		data_batch['correct_answer'] = []
		data_batch['bayesian_belief'] = []

		baseline = 0
		for tj in trajectory_batch:
			baseline += tj[0]['gain']
		baseline /= len(trajectory_batch)

		wait_debug = []
		while step_idx < max_length:
			for trajectory in trajectory_batch:
				if len(trajectory) > step_idx:# and trajectory[step_idx]['action'] != -1:
					data_batch['distractors'].append(trajectory[step_idx]['distractors'])
					data_batch['belief'].append(trajectory[step_idx]['belief'])
					data_batch['bayesian_belief'].append(trajectory[step_idx]['bayesian_belief'])
					data_batch['message'].append(trajectory[step_idx]['message'])
					data_batch['action'].append(trajectory[step_idx]['action'])
					data_batch['critic_value'].append(trajectory[step_idx]['gain'])
					data_batch['correct_answer'].append(trajectory[step_idx]['correct_answer'])
					data_batch['advantage'].append((trajectory[step_idx]['gain'] - baseline) / len(trajectory_batch))# - trajectory[step_idx]['critic_value'])

					if trajectory[step_idx]['action'] == self.num_distractors_ + 1:
						wait_debug.append({'advantage': data_batch['advantage'][-1],
										   'gain': trajectory[step_idx]['gain'],
										   'critic_value': trajectory[step_idx]['critic_value'],
										   'belief': trajectory[step_idx]['belief']})

			step_idx += 1

		data_batch['action'] = list(zip(range(len(data_batch['action'])), data_batch['action']))

		for k in data_batch:
			data_batch[k] = np.array(data_batch[k])
		data_batch['advantage'] /= num_traject

		if len(wait_debug) > 0:
			print('average wait advantage: %f' % np.mean([wa['advantage'] for wa in wait_debug]))
			#pdb.set_trace()


		loss_pg = -425
		if phase != 0:
			if phase == 1:
				train_op = self.policy_gradient_train_op_1_
			elif phase == 2:
				train_op = self.policy_gradient_train_op_2_
			elif phase == 3:
				train_op = self.policy_gradient_train_op_3_

			loss_pg, act_prob, bayesian_loss, _, boltzman_beta, belief_var_1d, act_prob, student_prediction, new_belief_var, msg_est_tensor_2d =\
				self.sess.run([self.loss_pg_, self.act_prob_, self.bayesian_violate_loss_, train_op, self.boltzman_beta_, self.belief_var_1d_, self.act_prob_, self.student_prediction_, self.new_belief_var_, self.msg_est_tensor_2d_],
									feed_dict = {self.distractors_: data_batch['distractors'],
									  			 self.student_belief_: data_batch['belief'],
									  			 self.bayesian_belief_spvs_: data_batch['bayesian_belief'],
									  			 self.teacher_belief_: data_batch['correct_answer'],
									  			 self.advantage_: np.squeeze(data_batch['advantage']),
									  			 self.prediction_indices_: data_batch['action'],
									  			 self.message_: data_batch['message']})
			#print('bayesian_loss: %f' % bayesian_loss)
			print('Stu: boltzman_beta: %f' % boltzman_beta)
			print('Stu: belief bias: %f' % belief_var_1d)
			#pdb.set_trace()
			# print('student_prediction', student_prediction)
			# print('act prob', act_prob)
			print('before normalization', np.mean(abs(msg_est_tensor_2d)))
		
		loss_vi, critic_value, distcts_feat_tensor_3, _ = self.sess.run([self.loss_vi_, self.critic_value_, self.distcts_feat_tensor_3_, self.value_iteration_train_op_],
								feed_dict = {self.distractors_: data_batch['distractors'],
											 self.teacher_belief_: data_batch['correct_answer'], 
											 self.student_belief_: data_batch['belief'],
											 self.critic_value_spvs_: data_batch['critic_value']})
		ridx = np.random.randint(data_batch['critic_value'].shape[0])
		print('critic value est:', critic_value[ridx: ridx + 10], 'critic value', data_batch['critic_value'][ridx: ridx + 10])
		#print(distcts_feat_tensor_3[ridx, :])
		return loss_pg, loss_vi

	def get_prediction_new_belief(self, distractors, prev_belief, embed_msg, correct_answer, student_think, student_wait):
		predict_prob, student_belief_new, critic_value, df_msg_match, df_msg_match_norm, new_belief_var, new_belief_var_process =\
			self.sess.run([self.student_prediction_, self.new_student_belief_full_norm_, self.critic_value_, self.df_msg_match_, self.df_msg_match_norm_, self.new_belief_var_, self.new_belief_var_process_],
													 					feed_dict = {self.distractors_: distractors,
													 					 			 self.student_belief_: prev_belief,
													 					 			 self.message_: embed_msg,
													 					 			 self.teacher_belief_: correct_answer})

		if student_wait:
			action_prob = predict_prob[0]
		else:
			action_prob = student_belief_new[0]

		if not student_think:
			try:
				action_idx = np.random.choice(self.num_distractors_ + 1 + student_wait, 1, p = action_prob)[0]
			except ValueError:
				pdb.set_trace()
		else:
			action_idx = np.argmax(action_prob)
		if np.sum(np.isnan(df_msg_match_norm)) != 0:
			pdb.set_trace()
		return action_idx, student_belief_new[:, 0: self.num_distractors_], predict_prob, critic_value, df_msg_match_norm

if __name__ == '__main__':
	main()
