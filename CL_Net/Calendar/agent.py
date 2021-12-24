import os
from collections import defaultdict
import tensorflow as tf
import numpy as np

import pdb

class Agent:
	def __init__(self, name, sess, num_slots):
		self.name_ = name
		self.sess = sess
		self.num_distractors_ = 2 ** num_slots

		self.num_slots_ = num_slots

		################
		# Placeholders #
		################
		with tf.variable_scope('%s_Agent' % self.name_, reuse = tf.AUTO_REUSE):
			self.calendar_tensor_ = tf.placeholder(tf.float32, shape = [None, self.num_distractors_, 1, self.num_slots_], name = 'calendars')
			self.message_ = tf.placeholder(tf.float32, shape = [None, self.num_slots_], name = 'message')
			self.other_belief_on_self_ = tf.placeholder(tf.float32, shape = [None, self.num_slots_], name = 'other_belief_on_self_')
			self.other_belief_on_self_spvs_ = tf.placeholder(tf.float32, shape = [None, self.num_slots_], name = 'other_belief_on_self_spvs_')
			self.self_belief_on_other_ = tf.placeholder(tf.float32, shape = [None, self.num_slots_], name = 'self_belief_on_other_')
			self.self_belief_on_self_ = tf.placeholder(tf.float32, shape = [None, self.num_slots_], name = 'self_belief_on_self_')

			self.q_net_spvs_ = tf.placeholder(tf.float32, shape = [None], name = 'q_net_spvs_')
			self.advantage_ = tf.placeholder(tf.float32, shape = [None])
			self.prediction_indices_ = tf.placeholder(tf.int32, shape = [None, 2])
			self.pretrain_belief_spvs_ = tf.placeholder(tf.float32, shape = [None, self.num_slots_], name = 'pretrain_self_belief_on_other')
			
			#actor-critic: critic
			self.other_belief_on_other_ = tf.placeholder(tf.float32, shape = [None, self.num_slots_], name = 'other_belief_on_other_')
			self.critic_value_spvs_ = tf.placeholder(tf.float32, shape = [None])

		with tf.variable_scope('%s_Distractor_Feature_Extraction' % self.name_, reuse = tf.AUTO_REUSE):
			self.df1_ = tf.layers.conv2d(self.calendar_tensor_, 3 * self.num_slots_, kernel_size = [1, 1],
								   		 kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2),
								   		 activation = tf.nn.leaky_relu)
			self.df2_ = tf.layers.conv2d(self.df1_, 2 * self.num_slots_, kernel_size = [1, 1],
								   		 kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2),
								   		 activation = tf.nn.leaky_relu)
			self.df3_ = tf.layers.conv2d(self.df2_, 1 * self.num_slots_, kernel_size = [1, 1],
								   		 kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2),
								   		 activation = tf.nn.leaky_relu)
		#######################
		#network belief update#
		#######################
		#teacher simulates student's new belief with all messages
		with tf.variable_scope('%s_Belief_OS_Predict' % self.name_, reuse = tf.AUTO_REUSE):
			# self.bos_feats_ = self.concat_cnn(self.df3_, [3 * self.num_slots_, 2 * self.num_slots_, self.num_slots_], [20, 10, self.num_distractors_])

			# self.bos_msg_compat_ = tf.reduce_sum(tf.multiply(tf.squeeze(self.bos_feats_[-1], axis = 2), tf.expand_dims(self.message_, 1)), axis = 2)

			# self.bos_belief_var_1d_ = tf.Variable(initial_value = -4, trainable = True, dtype = tf.float32)

			# self.bos_kernel_columns_ = []
			# for i in range(self.num_slots_):
			# 	kernel_fc1 = tf.contrib.layers.fully_connected(self.bos_msg_compat_, 3 * self.num_slots_, activation_fn = tf.nn.leaky_relu)
			# 	kernel_fc2 = tf.contrib.layers.fully_connected(kernel_fc1, 2 * self.num_slots_, activation_fn = tf.nn.leaky_relu)
			# 	kernel_fc3 = tf.contrib.layers.fully_connected(kernel_fc2, self.num_slots_, activation_fn = None)
			# 	self.bos_kernel_columns_.append(tf.expand_dims(kernel_fc3, -1))
			# self.bos_kernel_ = tf.concat(self.bos_kernel_columns_, axis = 2)

			# self.other_belief_on_self_pred_ = tf.squeeze(tf.sigmoid(tf.matmul(self.bos_kernel_, tf.expand_dims(self.other_belief_on_self_, -1)) + self.bos_belief_var_1d_), -1)

			self.msg_old_bos_ = tf.concat([self.message_, self.other_belief_on_self_], axis = 1)

			self.msg_old_bos_fc1_ = tf.contrib.layers.fully_connected(self.msg_old_bos_, 2 * self.num_slots_, activation_fn = tf.nn.leaky_relu)
			self.msg_old_bos_fc2_ = tf.contrib.layers.fully_connected(self.msg_old_bos_fc1_, 2 * self.num_slots_, activation_fn = tf.nn.leaky_relu)
			self.msg_old_bos_fc3_ = tf.contrib.layers.fully_connected(self.msg_old_bos_fc2_, 2 * self.num_slots_, activation_fn = None)
			self.other_belief_on_self_pred_ = tf.clip_by_value(tf.contrib.layers.fully_connected(self.msg_old_bos_fc3_, self.num_slots_, activation_fn = tf.exp), 0, 1)

			# self.bos_cross_entropy_ = -1 * tf.reduce_mean(tf.reduce_sum(tf.multiply(self.other_belief_on_self_spvs_, tf.math.log(self.other_belief_on_self_pred_ + 1e-9) +\
			# 															tf.multiply(1 - self.other_belief_on_self_spvs_, tf.math.log(1 - self.other_belief_on_self_pred_ + 1e-9))), axis = 1))

			self.bos_cross_entropy_ = tf.nn.l2_loss(self.other_belief_on_self_pred_ - self.other_belief_on_self_spvs_)
			
			self.bos_predict_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('%s_Belief_OS_Predict' % self.name_)]
			self.other_belief_on_self_update_opt_ = tf.train.AdamOptimizer(learning_rate = 1e-5)
			self.bos_predict_train_op_ = self.other_belief_on_self_update_opt_.minimize(self.bos_cross_entropy_, var_list = self.bos_predict_varlist_)

		####################
		# Q-Net Estimation #
		####################
		#check which new belief is the most ideal one, i think the reason for combine self_belief_on_other as input 
		# is to send message that can most probability get an agreement 
		with tf.variable_scope('%s_Q-Net' % self.name_, reuse = tf.AUTO_REUSE):
			self.q_net_opt_ = tf.train.AdamOptimizer(learning_rate = 1e-4)
			self.q_df_b1_ = tf.identity(self.self_belief_on_self_)
			self.q_df_b2_ = tf.identity(self.self_belief_on_other_)
			self.q_df_b3_ = tf.identity(self.other_belief_on_self_pred_)
			self.q_concat_df_b_ = tf.concat([self.q_df_b1_, self.q_df_b2_, self.q_df_b3_], axis = 1)
			
			self.q_dfb_merge_pre_1_ = tf.contrib.layers.fully_connected(self.q_concat_df_b_, 9, activation_fn = tf.nn.leaky_relu,
																weights_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-1))
			self.q_dfb_merge_pre_2_ = tf.contrib.layers.fully_connected(self.q_dfb_merge_pre_1_, 6, activation_fn = tf.nn.leaky_relu,
																weights_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-1))
			self.q_dfb_merge_pre_3_ = tf.contrib.layers.fully_connected(self.q_dfb_merge_pre_2_, 4, activation_fn = tf.nn.leaky_relu,
																weights_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-1))
			self.q_dfb_merge_ = tf.contrib.layers.fully_connected(self.q_dfb_merge_pre_3_, 1, activation_fn = None,
																weights_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2))
			self.q_value_ = tf.squeeze(self.q_dfb_merge_)

			self.q_net_loss_ = tf.reduce_mean(tf.square(self.q_value_ - self.q_net_spvs_))
			self.q_net_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('%s_Q-Net' % self.name_)]
			self.q_net_train_op_ = self.q_net_opt_.minimize(self.q_net_loss_, var_list = self.q_net_varlist_)

		#######################
		#network belief update#
		#######################
		with tf.variable_scope('%s_Belief_SO_Update' % self.name_, reuse = tf.AUTO_REUSE):
			# self.bso_feats_ = self.concat_cnn(self.df3_, [3 * self.num_slots_, 2 * self.num_slots_, self.num_slots_], [20, 10, self.num_distractors_])

			# self.bso_msg_compat_ = tf.reduce_sum(tf.multiply(tf.squeeze(self.bso_feats_[-1], axis = 2), tf.expand_dims(self.message_, 1)), axis = 2)

			# self.bso_belief_var_1d_ = tf.Variable(initial_value = -4, trainable = True, dtype = tf.float32)
			
			# self.bso_kernel_columns_ = []
			# for i in range(self.num_slots_):
			# 	kernel_fc1 = tf.contrib.layers.fully_connected(self.bos_msg_compat_, 3 * self.num_slots_, activation_fn = tf.nn.leaky_relu)
			# 	kernel_fc2 = tf.contrib.layers.fully_connected(kernel_fc1, 2 * self.num_slots_, activation_fn = tf.nn.leaky_relu)
			# 	kernel_fc3 = tf.contrib.layers.fully_connected(kernel_fc2, self.num_slots_, activation_fn = None)
			# 	self.bso_kernel_columns_.append(tf.expand_dims(kernel_fc3, -1))
			# self.bso_kernel_ = tf.concat(self.bso_kernel_columns_, axis = 2)

			self.msg_old_bso_ = tf.concat([self.message_, self.self_belief_on_other_], axis = 1)

			self.msg_old_bso_fc1_ = tf.contrib.layers.fully_connected(self.msg_old_bso_, 2 * self.num_slots_, activation_fn = tf.nn.leaky_relu)
			self.msg_old_bso_fc2_ = tf.contrib.layers.fully_connected(self.msg_old_bso_fc1_, 2 * self.num_slots_, activation_fn = tf.nn.leaky_relu)
			self.msg_old_bso_fc3_ = tf.contrib.layers.fully_connected(self.msg_old_bso_fc2_, 2 * self.num_slots_, activation_fn = None)
			self.new_self_belief_on_other_ = tf.clip_by_value(tf.contrib.layers.fully_connected(self.msg_old_bso_fc3_, self.num_slots_, activation_fn = tf.exp), 0, 1)

			## pretrain varaibles ##
			# self.bso_cross_entropy_ = -1 * tf.reduce_mean(tf.reduce_sum(tf.multiply(self.pretrain_belief_spvs_,
			# 																		tf.math.log(self.new_self_belief_on_other_ + 1e-9)) +\
			# 															tf.multiply(1 - self.pretrain_belief_spvs_,
			# 																		tf.math.log(1 - self.new_self_belief_on_other_ + 1e-9)), axis = 1))
			self.bso_cross_entropy_ = tf.reduce_mean(tf.nn.l2_loss(self.new_self_belief_on_other_ - self.pretrain_belief_spvs_))
			
			self.pretrain_bso_opt_ = tf.train.AdamOptimizer(learning_rate = 1e-5)
			self.pretrain_bso_train_op_ = self.pretrain_bso_opt_.minimize(self.bso_cross_entropy_)
			self.pretrain_bso_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)\
									  if v.name.startswith('%s_Belief_SO_Update' % self.name_)\
									  or v.name.startswith('%s_Distractor_Feature_Extraction' % self.name_)]

			self.pretrain_op_ = tf.group(self.bos_predict_train_op_, self.pretrain_bso_train_op_)					  
			self.pretrain_saver_ = tf.train.Saver()
			self.pretrain_loader_ = tf.train.Saver([v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('%s_Belief' % self.name_)])
		
		####################
		# V-Net Estimation #
		####################
		with tf.variable_scope('%s_Value_Network' % self.name_, reuse = tf.AUTO_REUSE):
			#this network is only to tune belief update as a listener, so state only involves the following three beliefs
			self.v_df_b1_ = tf.identity(self.self_belief_on_self_)
			self.v_df_b2_ = tf.identity(self.other_belief_on_other_)
			self.v_df_b3_ = tf.identity(self.self_belief_on_other_)
			self.v_concat_df_b_ = tf.concat([self.v_df_b1_, self.v_df_b2_, self.v_df_b3_], axis = 1)

			self.v_dfb_merge_pre_1_ = tf.contrib.layers.fully_connected(self.v_concat_df_b_, 9, activation_fn = tf.nn.leaky_relu,
																weights_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-1))
			self.v_dfb_merge_pre_2_ = tf.contrib.layers.fully_connected(self.v_dfb_merge_pre_1_, 6, activation_fn = tf.nn.leaky_relu,
																weights_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-1))
			self.v_dfb_merge_pre_3_ = tf.contrib.layers.fully_connected(self.v_dfb_merge_pre_2_, 4, activation_fn = tf.nn.leaky_relu,
																weights_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-1))
			self.v_dfb_merge_ = tf.contrib.layers.fully_connected(self.v_dfb_merge_pre_3_, 1, activation_fn = None,
																weights_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2))
			self.critic_value_ = tf.squeeze(self.v_dfb_merge_)

			self.v_net_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)\
							if v.name.startswith('%s_Value_Network' % self.name_)]

			self.v_net_loss_ = tf.reduce_mean(tf.square(self.critic_value_ - self.critic_value_spvs_))

			self.value_iteration_opt_ = tf.train.AdamOptimizer(learning_rate = 1e-4)
			self.value_iteration_train_op_ = self.value_iteration_opt_.minimize(self.v_net_loss_, var_list = self.v_net_varlist_)

		####################
		#  Policy Network  #
		####################
		with tf.variable_scope('%s_Policy_Network' % self.name_, reuse = tf.AUTO_REUSE):
			self.common_slots_ = tf.multiply(1 - self.self_belief_on_self_, 1 - self.new_self_belief_on_other_)
			#self.common_slots_ = tf.concat([self.self_belief_on_self_, self.new_self_belief_on_other_], axis = 1)
			
			self.reject_prob_1_ = tf.contrib.layers.fully_connected(self.common_slots_, self.num_slots_,
																	  activation_fn = tf.nn.leaky_relu)
			self.reject_prob_2_ = tf.contrib.layers.fully_connected(self.reject_prob_1_, self.num_slots_,
																	  activation_fn = tf.nn.leaky_relu)																	  
			self.reject_prob_3_ = tf.contrib.layers.fully_connected(self.reject_prob_1_, 4,
																	  activation_fn = tf.nn.leaky_relu)
			self.reject_prob_ = tf.contrib.layers.fully_connected(self.reject_prob_3_, 1, activation_fn = tf.nn.sigmoid)

			self.hold_prob_1_ = tf.contrib.layers.fully_connected(self.common_slots_, self.num_slots_,
																	  activation_fn = tf.nn.leaky_relu)
			self.hold_prob_2_ = tf.contrib.layers.fully_connected(self.hold_prob_1_, self.num_slots_,
																	  activation_fn = tf.nn.leaky_relu)
			self.hold_prob_3_ = tf.contrib.layers.fully_connected(self.hold_prob_2_, 4,
																	  activation_fn = tf.nn.leaky_relu)
			self.hold_prob_ = tf.contrib.layers.fully_connected(self.hold_prob_3_, 1, activation_fn = tf.nn.sigmoid)
			
			self.decision_prob_1_ = tf.contrib.layers.fully_connected(self.common_slots_, 2 * self.num_slots_,
																	  activation_fn = tf.nn.leaky_relu)
			self.decision_prob_2_ = tf.contrib.layers.fully_connected(self.decision_prob_1_, 2 * self.num_slots_,
																	  activation_fn = tf.nn.leaky_relu)
			self.decision_prob_3_ = tf.contrib.layers.fully_connected(self.decision_prob_2_, 2 * self.num_slots_,
																	  activation_fn = tf.nn.leaky_relu)
			self.decision_prob_ = tf.contrib.layers.fully_connected(self.decision_prob_3_, self.num_slots_,
																	activation_fn = tf.nn.softmax)

			self.action_prob_ = tf.concat([self.decision_prob_, self.hold_prob_, self.reject_prob_], axis = 1)
			#self.action_prob_ = self.decision_prob_
			self.pg_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)\
								if v.name.startswith('%s_Policy_Network' % self.name_)]
			
			self.bn_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)\
								if v.name.startswith('%s_Belief' % self.name_)]

			self.policy_regularization_ = tf.add_n([ tf.nn.l2_loss(v) for v in self.pg_varlist_ if 'bias' not in v.name ])
			
			self.act_prob_ = tf.math.log(tf.gather_nd(self.action_prob_, self.prediction_indices_) + 1e-9)
			self.not_hold_prob_ = tf.math.log(tf.where(self.prediction_indices_[:, 1: 2] < self.num_slots_, 1 - self.hold_prob_ + 1e-9, tf.ones_like(self.hold_prob_)))
			self.not_hold_prob_ += tf.math.log(tf.where(self.prediction_indices_[:, 1: 2] == self.num_slots_ + 1, 1 - self.hold_prob_ + 1e-9, tf.ones_like(self.hold_prob_)))
			self.not_reject_prob_ = tf.math.log(tf.where(self.prediction_indices_[:, 1: 2] < self.num_slots_, 1 - self.reject_prob_ + 1e-9, tf.ones_like(self.reject_prob_)))
			
			self.total_log_ = self.act_prob_ + self.not_hold_prob_ + self.not_reject_prob_
			
			self.decision_entropy_ = tf.reduce_sum(tf.log(self.decision_prob_ + 1e-9), axis = 1)
			self.loss_pg_ = -1 * tf.reduce_sum(tf.multiply(self.total_log_, self.advantage_)) + 1e-4 * self.decision_entropy_ + 1e-4 * self.policy_regularization_

			self.policy_gradient_opt_ = tf.train.AdamOptimizer(learning_rate = 1e-4)
			self.policy_gradient_behave_train_op_ = self.policy_gradient_opt_.minimize(self.loss_pg_, var_list = self.pg_varlist_)
			self.policy_gradient_belief_train_op_ = self.policy_gradient_opt_.minimize(self.loss_pg_, var_list = self.bn_varlist_)
			self.policy_gradient_train_op_ = self.policy_gradient_opt_.minimize(self.loss_pg_, var_list = self.bn_varlist_ + self.pg_varlist_)

			self.total_saver_ = tf.train.Saver()
			self.total_loader_ = tf.train.Saver([v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith(self.name_)])


	def concat_cnn(self, input_tensor, num_filters, num_dims):
		assert(num_dims[-1] == self.num_distractors_)
		feat_tensors = [input_tensor]
		for idx, num_filter in enumerate(num_filters):
			tensor_rows = []
			kernel_dim = self.num_distractors_ if idx == 0 else num_dims[idx - 1]
			for _ in range(num_dims[idx]):
				tensor_rows.append(tf.layers.conv2d(feat_tensors[-1], num_filter, kernel_size = [kernel_dim, 1],
														  kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2),
														  padding = 'valid', activation = tf.nn.leaky_relu))

			feat_tensor = tf.concat(tensor_rows, axis = 1)
			feat_tensors.append(feat_tensor)

		return feat_tensors
	
	def pretrain_belief_update(self, data_batch):
		_, bso_cross_entropy, bos_cross_entropy, new_self_belief_on_other, other_belief_on_self_pred =\
			self.sess.run([self.pretrain_op_, self.bso_cross_entropy_, self.bos_cross_entropy_,
						   self.new_self_belief_on_other_, self.other_belief_on_self_pred_],
						  feed_dict = {self.self_belief_on_other_: data_batch['prev_belief'],
									   self.message_: data_batch['message'],
									   self.calendar_tensor_: data_batch['distractors'],
									   self.pretrain_belief_spvs_: data_batch['new_belief'],
									   self.other_belief_on_self_: data_batch['prev_belief'],
									   self.other_belief_on_self_spvs_: data_batch['new_belief']})
		
		if np.sum(np.isnan(other_belief_on_self_pred)) > 0 or np.sum(np.isnan(new_self_belief_on_other)) > 0:
			pdb.set_trace()
		if np.isnan(bso_cross_entropy) or np.isnan(bos_cross_entropy):
			pdb.set_trace()

		return bso_cross_entropy, bos_cross_entropy, new_self_belief_on_other, other_belief_on_self_pred

	def train_belief_update(self, data_batch):
		_, bos_cross_entropy, other_belief_on_self_pred =\
			self.sess.run([self.bos_predict_train_op_, self.bos_cross_entropy_, self.other_belief_on_self_pred_],
						  feed_dict = {self.message_: data_batch['message'],
									   self.calendar_tensor_: data_batch['distractors'],
									   self.other_belief_on_self_: data_batch['prev_belief'],
									   self.other_belief_on_self_spvs_: data_batch['new_belief']})
		
		if np.sum(np.isnan(other_belief_on_self_pred)) > 0:
			pdb.set_trace()
		if np.isnan(bos_cross_entropy):
			pdb.set_trace()

		return bos_cross_entropy, other_belief_on_self_pred
	
	def train_q_net(self, data_batch):
		_, q_net_loss, q_value = self.sess.run([self.q_net_train_op_, self.q_net_loss_, self.q_value_],\
									  feed_dict = {self.calendar_tensor_: data_batch['distractors'],
									  			   self.q_net_spvs_: data_batch['target_q'],
												   self.message_: data_batch['message'],
									  			   self.self_belief_on_self_: data_batch['self_belief_on_self'],
						   			   			   self.self_belief_on_other_: data_batch['self_belief_on_other'],
												   self.other_belief_on_self_: data_batch['other_belief_on_self']})
		ridx = np.random.randint(q_value.shape[0])
		print('%s value est:' % self.name_, q_value[ridx: ridx + 10], data_batch['target_q'][ridx: ridx + 10])
		print('Q learning loss: %f' % q_net_loss)

		return q_net_loss

	def pretrain_bayesian_belief_update(self, concept_generator, agent_pretraining_steps, agent_pretrain_batch_size,
										agent_pretrain_ckpt_dir, agent_pretrain_ckpt_name, continue_steps = 0, silent = False):
		if not os.path.exists(agent_pretrain_ckpt_dir):
			os.makedirs(agent_pretrain_ckpt_dir)

		ckpt_dir = os.path.join(agent_pretrain_ckpt_dir, self.name_)
		if not os.path.exists(ckpt_dir):
			os.makedirs(ckpt_dir)
		
		ckpt = tf.train.get_checkpoint_state(ckpt_dir)
		train_steps = agent_pretraining_steps
		if ckpt:
			self.pretrain_loader_.restore(self.sess, ckpt.model_checkpoint_path)
			print('Loaded agent %s belief update ckpt from %s' % (self.name_, ckpt_dir))
			train_steps = continue_steps
			
		accuracies = []
		l1_diffs = []
		bayesian_wrongs = []
		cross_entropies = []
		for ts in range(train_steps):
			data_batch = concept_generator.generate_batch(agent_pretrain_batch_size)
			bso_cross_entropy, bos_cross_entropy, belief_pred_1, belief_pred_2 = self.pretrain_belief_update(data_batch)			
			
			belief_pred = np.concatenate([belief_pred_1, belief_pred_2], axis = 0)
			target = np.tile(data_batch['new_belief'], [2, 1])
			l1_diff = abs(belief_pred - target)
			correct = np.mean(l1_diff <= 1e-2)
			bayesian_wrong = np.mean(np.sum((target == 0) * (belief_pred > 1e-5), axis = 1) > 0)
			accuracies.append(np.mean(correct))
			l1_diffs.append(np.mean(l1_diff))
			cross_entropies.append(0.5 * (bso_cross_entropy + bos_cross_entropy))
			bayesian_wrongs.append(bayesian_wrong)
			if ts % 1000 == 0 and not silent:
				print('[%s:%d] batch mean cross entropy: %f, mean accuracies: %f, mean l1: %f, bayesian wrong: %f'\
					  % (self.name_, ts + 1, np.mean(cross_entropies), np.mean(accuracies), np.mean(l1_diffs), np.mean(bayesian_wrongs)))
				if np.mean(accuracies) >= 0:
					idx = np.random.randint(2 * agent_pretrain_batch_size)
					print('\t target:', target[idx, :])
					print('\t predict', belief_pred[idx, :])
				accuracies = []
				l1_diffs = []
				bayesian_wrongs = []
			if (ts + 1) % 10000 == 0:
				self.pretrain_saver_.save(self.sess, os.path.join(ckpt_dir, agent_pretrain_ckpt_name),
											  global_step = agent_pretraining_steps)
				print('Saved agent %s belief update ckpt to %s after %d training'\
					  % (self.name_, ckpt_dir, ts))
		if train_steps != 0:
			self.pretrain_saver_.save(self.sess, os.path.join(ckpt_dir, agent_pretrain_ckpt_name),
											  global_step = agent_pretraining_steps)
			print('Saved agent %s belief update ckpt to %s after %d training'\
				  % (self.name_, ckpt_dir, train_steps))

	#copy current belief for all messages, get q value for all message-belief pair
	def get_q_value_for_all_msg(self, self_belief_on_self, self_belief_on_other, other_belief_on_self, concept_generator):
		all_msg_embeddings = concept_generator.all_msgs_tensor_
		num_total_msgs = concept_generator.num_msgs_
		bss_tile = np.tile(self_belief_on_self, (num_total_msgs, 1))
		bso_tile = np.tile(self_belief_on_other, (num_total_msgs, 1))
		bos_tile = np.tile(other_belief_on_self, (num_total_msgs, 1))
		calendars_tile = np.tile(np.expand_dims(
			np.expand_dims(concept_generator.tensor_, axis = 1), axis = 0), (num_total_msgs, 1, 1, 1))
		q_values, other_belief_on_self_pred =\
			self.sess.run([self.q_value_, self.other_belief_on_self_pred_],
						  feed_dict = {self.calendar_tensor_: calendars_tile,
									   self.message_: all_msg_embeddings,
									   self.self_belief_on_self_: bss_tile,
									   self.self_belief_on_other_: bso_tile,
									   self.other_belief_on_self_: bos_tile})

		return q_values, other_belief_on_self_pred
	
	#first listener update self belief on other according to message passed on
	#then use the new belief on other to generate action
	def update_self_belief_on_other(self, prev_self_belief_on_other, other_belief_on_other,
									self_belief_on_self, embed_msg, concept_generator, is_training):
		calendars_tile = np.expand_dims(np.expand_dims(concept_generator.tensor_, axis = 1), axis = 0)
		decision_prob, hold_prob, reject_prob, slots_belief, predict_prob, critic_value =\
			self.sess.run([self.decision_prob_, self.hold_prob_, self.reject_prob_, self.common_slots_, self.new_self_belief_on_other_, self.critic_value_],
						  feed_dict = {self.calendar_tensor_: calendars_tile,
						  			   self.self_belief_on_other_: prev_self_belief_on_other,
									   self.self_belief_on_self_: self_belief_on_self,
									   self.other_belief_on_other_: other_belief_on_other,
									   self.message_: embed_msg})
		action_prob = np.concatenate([decision_prob, hold_prob, reject_prob], axis = 1)
		if np.sum(np.isnan(reject_prob)) > 0 or np.sum(np.isnan(hold_prob)) > 0:
			pdb.set_trace()
		if is_training:
			hold = np.random.choice(2, 1, p = [1 - hold_prob[0][0], hold_prob[0][0]])[0]
			if hold:
				action_idx = self.num_slots_
			else:
				reject = np.random.choice(2, 1, p = [1 - reject_prob[0][0], reject_prob[0][0]])[0]
				if reject:
					action_idx = self.num_slots_ + 1
				else:
					try:
						action_idx = np.random.choice(self.num_slots_, 1, p = decision_prob[0])[0]
					except ValueError:
						pdb.set_trace()
		else:
			if hold_prob[0][0] > 0.5:
				action_idx = self.num_slots_
			elif reject_prob[0][0] > 0.5:
				action_idx = self.num_slots_ + 1
			else:
				action_idx = np.argmax(decision_prob)
		return action_idx, slots_belief, predict_prob, action_prob, critic_value

	# trajectory_batch = [[(distractors, belief, msg, action, advantage), ...], [], []]
	# advantage should include 1/m, where m is the batch size
	def update_net(self, trajectory_batch, phase = 1):
		debug_batch = {}

		#listener updates
		num_traject = len(trajectory_batch['trajectory_batch'])
		max_length = max([len(traject) for traject in trajectory_batch])
		step_idx = 0
		data_batch = {}

		data_batch['distractors'] = []
		data_batch['other_belief_on_other'] = []
		data_batch['self_belief_on_other'] = []
		data_batch['self_belief_on_self'] = []
		data_batch['message'] = []
		data_batch['action'] = []
		data_batch['advantage'] = []
		data_batch['critic_value'] = []
		data_batch['correct_answer'] = []
		data_batch['bayesian_belief'] = []

		baseline = 0
		for tj in trajectory_batch['trajectory_batch']:
			baseline += tj[0]['gain']
		baseline /= len(trajectory_batch['trajectory_batch'])
		wait_debug = []
		while step_idx < max_length:
			for trajectory in trajectory_batch['trajectory_batch']:
				if len(trajectory) > step_idx:
					data_batch['distractors'].append(trajectory[step_idx]['distractors'])
					data_batch['other_belief_on_other'].append(trajectory[step_idx]['other_belief_on_other'])
					data_batch['self_belief_on_other'].append(trajectory[step_idx]['self_belief_on_other'])
					data_batch['self_belief_on_self'].append(trajectory[step_idx]['self_belief_on_self'])
					data_batch['message'].append(trajectory[step_idx]['message'])
					data_batch['action'].append(trajectory[step_idx]['action'])
					data_batch['critic_value'].append(trajectory[step_idx]['gain'])
					data_batch['advantage'].append((trajectory[step_idx]['gain'] - baseline) / len(trajectory_batch['trajectory_batch']))

					if trajectory[step_idx]['action'] == self.num_slots_:
						wait_debug.append({'advantage': data_batch['advantage'][-1],
										   'gain': trajectory[step_idx]['gain'],
										   'critic_value': trajectory[step_idx]['critic_value']})

			step_idx += 1

		data_batch['action'] = list(zip(range(len(data_batch['action'])), data_batch['action']))

		for k in data_batch:
			data_batch[k] = np.array(data_batch[k])
		data_batch['advantage'] /= num_traject

		if len(wait_debug) > 0:
			print('average wait advantage: %f' % np.mean([wa['advantage'] for wa in wait_debug]))
			#pdb.set_trace()


		loss_pg = -425
		
		if phase == 1 or phase == 0:
			train_op = self.policy_gradient_behave_train_op_
		elif phase == 2:
			train_op = self.policy_gradient_belief_train_op_
		elif phase == 3:
			train_op = self.policy_gradient_behave_train_op_
		
		prev_weights = self.sess.run(self.pg_varlist_)
		# loss_pg, _, action_prob,\
		# decision_prob_1, decision_prob_2, decision_prob_3,\
		# act_prob, not_hold_prob, not_reject_prob =\
		# 	self.sess.run([self.loss_pg_, train_op, self.action_prob_,
		# 				   self.decision_prob_1_, self.decision_prob_2_, self.decision_prob_3_,
		# 				   self.act_prob_, self.not_hold_prob_, self.not_reject_prob_],
		loss_pg, _, action_prob,\
		decision_prob_1, decision_prob_2, decision_prob_3 =\
			self.sess.run([self.loss_pg_, train_op, self.action_prob_,
						   self.decision_prob_1_, self.decision_prob_2_, self.decision_prob_3_],
								feed_dict = {self.calendar_tensor_: data_batch['distractors'],
												self.other_belief_on_other_: data_batch['other_belief_on_other'], 
												self.self_belief_on_self_: data_batch['self_belief_on_self'],
												self.self_belief_on_other_: data_batch['self_belief_on_other'],
												self.advantage_: data_batch['advantage'],
												self.prediction_indices_: data_batch['action'],
												self.message_: data_batch['message']})
		
		new_weights = self.sess.run(self.pg_varlist_)
		for nw in new_weights:
			if np.sum(np.isnan(nw)) > 0 or np.sum(np.isinf(nw)) > 0:
				pdb.set_trace()	
		
		v_net_loss, _ = self.sess.run([self.v_net_loss_, self.value_iteration_train_op_],
								feed_dict = {self.calendar_tensor_: data_batch['distractors'],
											 self.other_belief_on_other_: data_batch['other_belief_on_other'], 
											 self.self_belief_on_self_: data_batch['self_belief_on_self'],
											 self.self_belief_on_other_: data_batch['self_belief_on_other'],
											 self.critic_value_spvs_: data_batch['critic_value']})

		debug_batch['v_net_loss'] = v_net_loss
		debug_batch['loss_pg'] = loss_pg

		if np.sum(np.isnan(action_prob)) > 0 or np.sum(np.isinf(action_prob)) > 0:
			pdb.set_trace()

		#speaker section
		belief_update_batch = {}
		belief_update_batch['prev_belief'] = []
		belief_update_batch['new_belief'] = []
		belief_update_batch['message'] = []
		belief_update_batch['distractors'] = []
		for belief_tuple in trajectory_batch['belief_update_batch']:
			belief_update_batch['distractors'].append(belief_tuple[0])
			belief_update_batch['prev_belief'].append(belief_tuple[1])
			belief_update_batch['message'].append(belief_tuple[2])
			belief_update_batch['new_belief'].append(belief_tuple[3])

		for k in belief_update_batch:
			belief_update_batch[k] = np.array(belief_update_batch[k])
		if phase == 1:
			cross_entropy, belief_pred = self.train_belief_update(belief_update_batch)
			print('%s\'s belief esimate cross_entropy: %f' % (self.name_, cross_entropy))

			debug_batch['%s_belief_prediction' % self.name_] = belief_pred

		q_learning_batch = {}
		q_learning_batch['self_belief_on_self'] = []
		q_learning_batch['self_belief_on_other'] = []
		q_learning_batch['other_belief_on_self'] = []
		q_learning_batch['message'] = []
		q_learning_batch['distractors'] = []
		q_learning_batch['target_q'] = []
		for q_learning_tuple in trajectory_batch['q_learning_batch']:
			q_learning_batch['distractors'].append(q_learning_tuple['distractors'])
			q_learning_batch['self_belief_on_self'].append(q_learning_tuple['self_belief_on_self'])
			q_learning_batch['self_belief_on_other'].append(q_learning_tuple['self_belief_on_other'])
			q_learning_batch['other_belief_on_self'].append(q_learning_tuple['other_belief_on_self'])
			q_learning_batch['message'].append(q_learning_tuple['message'])
			q_learning_batch['target_q'].append(q_learning_tuple['target_q'])

		for k in q_learning_batch:
			q_learning_batch[k] = np.array(q_learning_batch[k])
		if phase == 1:
			q_net_loss = self.train_q_net(q_learning_batch)

		return debug_batch

def main():
	num_slots = 8

	from concept import Concept
	calendars = Concept(num_slots)
	sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True,
	   										  log_device_placement = False))

	with tf.device('/cpu:0'):
		agent = Agent('A', sess, num_slots)

if __name__ == '__main__':
	main()