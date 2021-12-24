import os
import sys

import tensorflow as tf
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from agent import Agent
import pdb

class Student(Agent):
	def __init__(self, sess, train_config, game_config, visual_config, belief_config, value_config):
		Agent.__init__(self, sess, train_config, game_config, visual_config, belief_config, value_config, 'Student')

		self.no_op_ = tf.no_op()

		with tf.variable_scope('Student_Policy_Network'):
			self.advantage_ = tf.placeholder(tf.float32, shape = [None], name = 'advantage')
			self.prediction_indices_ = tf.placeholder(tf.int32, shape = [None, 2])
			self.new_student_belief_full_norm_ = self.meditation_core_.posterior_full_norm_
			
			#use entropy to delay final prediction
			self.new_belief_var_ = tf.reduce_sum(-1 * self.new_student_belief_full_norm_ *\
												 tf.math.log(self.new_student_belief_full_norm_ + 1e-9), axis = 1, keepdims = True)
			self.var_process_factor_1_ = tf.Variable(initial_value = 1, trainable = True, dtype = tf.float32)
			self.var_process_factor_2_ = tf.Variable(initial_value = 0, trainable = True, dtype = tf.float32)
			self.var_process_factor_3_ = tf.Variable(initial_value = 0, trainable = True, dtype = tf.float32)
			self.new_belief_var_process_ = tf.exp(self.var_process_factor_1_ * self.new_belief_var_) +\
												  self.var_process_factor_2_ * self.new_belief_var_ + self.var_process_factor_3_
			self.pre_prediction_ = tf.concat([self.new_student_belief_full_norm_, tf.nn.relu(self.new_belief_var_process_)], axis = 1)
			
			self.student_prediction_ = tf.div_no_nan(self.pre_prediction_, tf.reduce_sum(self.pre_prediction_, axis = 1, keepdims = True))

		## policy gradient ##
		self.pg_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)\
							if v.name.startswith('Student_Policy_Network')]

		self.act_prob_ = tf.math.log(tf.gather_nd(self.student_prediction_, self.prediction_indices_))
		self.seperate_loss = tf.multiply(self.act_prob_, self.advantage_)
		self.loss_pg_ = -1 * tf.reduce_sum(self.seperate_loss) + 1e-3 * self.regularization_

		self.policy_gradient_opt_ = tf.train.AdamOptimizer(learning_rate = 5e-4)
		self.policy_gradient_opt_finer_ = tf.train.AdamOptimizer(learning_rate = 5e-4)
		
		self.policy_gradient_train_op_1_ = self.policy_gradient_opt_finer_.minimize(self.loss_pg_, var_list = self.belief_varlist_)
		self.policy_gradient_train_op_2_ = self.policy_gradient_opt_.minimize(self.loss_pg_, var_list = self.pg_varlist_)

		self.total_loader_ = tf.train.Saver(list(set([v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)\
											if 'Adam' not in v.name and v.name.startswith('%s' % self.role_)] + self.batch_norm_varlist_)))
		self.total_saver_ = tf.train.Saver([v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)\
											if v.name.startswith('%s' % self.role_)], max_to_keep = None)

	# trajectory_batch = [[(distractors, belief, msg, action, advantage), ...], [], []]
	# advantage should include 1/m, where m is the batch size
	def update_net(self, trajectory_batch, phase = 1):

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

		baseline = 0
		for tj in trajectory_batch:
			baseline += tj[0]['gain']
		baseline /= num_traject
		wait_debug = []
		while step_idx < max_length:
			for trajectory in trajectory_batch:
				if len(trajectory) > step_idx:
					data_batch['distractors'].append(trajectory[step_idx]['distractors'])
					data_batch['belief'].append(trajectory[step_idx]['belief'])
					data_batch['message'].append(trajectory[step_idx]['message'])
					data_batch['action'].append(trajectory[step_idx]['action'])
					data_batch['critic_value'].append(trajectory[step_idx]['gain'])
					data_batch['correct_answer'].append(trajectory[step_idx]['correct_answer'])
					data_batch['advantage'].append((trajectory[step_idx]['gain'] - baseline) / num_traject)# - trajectory[step_idx]['critic_value'])

					if trajectory[step_idx]['action'] == self.num_distractors_ + 1:
						wait_debug.append({'advantage': data_batch['advantage'][-1],
										   'gain': trajectory[step_idx]['gain'],
										   'critic_value': trajectory[step_idx]['critic_value'],
										   'belief': trajectory[step_idx]['belief']})

			step_idx += 1

		data_batch['action'] = list(zip(range(len(data_batch['action'])), data_batch['action']))

		for k in data_batch:
			data_batch[k] = np.array(data_batch[k])
		# data_batch['advantage'] /= num_traject 
		if len(wait_debug) > 0:
			print('average wait advantage: %f' % np.mean([wa['advantage'] for wa in wait_debug]))
			#pdb.set_trace()


		loss_pg = -425
		if phase != 0:
			if phase == 1:
				train_op = self.policy_gradient_train_op_1_
			elif phase == 2:
				train_op = self.policy_gradient_train_op_2_

			boltzman_beta_op = self.meditation_core_.boltzman_beta_ if self.belief_config_.type == 'Explicit' else self.no_op_
			loss_pg, seperate_loss, act_prob, _, boltzman_beta, belief_var_1d =\
				self.sess_.run([self.loss_pg_, self.seperate_loss, self.act_prob_, train_op, boltzman_beta_op,
								self.meditation_core_.belief_var_1d_],
									feed_dict = {self.distractors_: data_batch['distractors'],
									  			 self.student_belief_: data_batch['belief'],
									  			 self.teacher_belief_: data_batch['correct_answer'],
									  			 self.advantage_: np.squeeze(data_batch['advantage']),
									  			 self.prediction_indices_: data_batch['action'],
									  			 self.message_: data_batch['message']})
			# print('Stu: boltzman_beta', boltzman_beta)
			# print('Stu: belief bias: %f' % belief_var_1d)

		return loss_pg

	def get_prediction_new_belief(self, distractors, prev_belief, embed_msg, correct_answer, student_think, student_wait, msg_as_belief = False):
		msg_likelihood_op = self.no_op_ if self.belief_config_.type == 'Implicit' else self.meditation_core_.compatibility_
		predict_prob, student_belief_new, critic_value, msg_likelihood =\
			self.sess_.run([self.student_prediction_,
							self.new_student_belief_full_norm_,
							self.value_core_.value_, msg_likelihood_op],
						   feed_dict = {self.distractors_: distractors,\
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
		
		if msg_as_belief:
			student_belief_new = np.zeros((1, self.num_distractors_ + 1 + student_wait))
			student_belief_new[0, action_idx] = 1

		return action_idx, student_belief_new[0, 0: self.num_distractors_], predict_prob, critic_value, msg_likelihood

if __name__ == '__main__':
	main()
