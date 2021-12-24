import os
import math
import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, os.path.dirname(__file__))
from agent import Agent

import pdb

#np.set_printoptions(threshold=np.nan, precision=4, suppress=True)

class Teacher(Agent):
	def __init__(self, sess, train_config, game_config, visual_config, belief_config, value_config):
		Agent.__init__(self, sess, train_config, game_config, visual_config, belief_config, value_config, 'Teacher')
		self.total_loader_ = tf.train.Saver(list(set([v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)\
											if 'Adam' not in v.name and v.name.startswith('%s' % self.role_)] + self.batch_norm_varlist_)))
		self.total_saver_ = tf.train.Saver([v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)\
											if v.name.startswith('%s' % self.role_)], max_to_keep = None)

	def train_q_net(self, data_batch):
		_, q_net_loss, value = self.sess_.run([self.value_train_op_, self.value_net_loss_, self.value_core_.value_],\
									  feed_dict = {self.value_spvs_: data_batch['target_q'],
									  			   self.student_belief_: data_batch['student_belief'],
						   			   			   self.message_: data_batch['message'],
						   			   			   self.distractors_: data_batch['distractors'],
						   			   			   self.teacher_belief_: data_batch['teacher_belief']})
		# print('Q learning loss: %f' % q_net_loss)
		ridx = np.random.randint(value.shape[0])
		#print(value[ridx], data_batch['target_q'][ridx])
		print('0.8: %f, 0.2: %f' % (np.sum(value * (data_batch['target_q'] == 0.8)) / np.sum(data_batch['target_q'] == 0.8),
									np.sum(value * (data_batch['target_q'] == -0.2)) / np.sum(data_batch['target_q'] == -0.2)))
		# print('Teacher value est:', value[ridx: ridx + 10], data_batch['target_q'][ridx: ridx + 10])
		#print(distcts_feat_tensor_3[ridx, :])

		return q_net_loss

	def get_q_value_for_all_msg(self, teacher_belief, student_belief, embeded_concepts):
		
		all_msg_embeddings = np.identity(self.message_space_size_)
		teacher_belief_tile = np.tile(teacher_belief, (self.message_space_size_, 1))
		student_belief_tile = np.tile(student_belief, (self.message_space_size_, 1))
		if self.visual_config_.type == 'Feat':
			embeded_concepts_tile = np.tile(embeded_concepts, (self.message_space_size_, 1, 1))
		else:
			embeded_concepts_tile = np.tile(embeded_concepts, (self.message_space_size_, 1, 1, 1, 1))
		q_values, belief_pred = self.sess_.run([self.value_core_.value_, self.meditation_core_.posterior_],
											feed_dict = {self.distractors_: embeded_concepts_tile,
														self.message_: all_msg_embeddings,
														self.teacher_belief_: teacher_belief_tile,
														self.student_belief_: student_belief_tile})
		# mingle1, mingle2, visual_feat, q_values, belief_pred = self.sess_.run([self.meditation_core_.mingle1_, self.meditation_core_.mingle2_, self.meditation_core_.visual_feat_, \
		# 										self.value_core_.value_, self.meditation_core_.posterior_],
		# 									feed_dict = {self.distractors_: embeded_concepts_tile,
		# 												self.message_: all_msg_embeddings,
		# 												self.teacher_belief_: teacher_belief_tile,
		# 												self.student_belief_: student_belief_tile})
		# print('mingle1: ', mingle1)
		# print('mingle2: ', mingle2)
		# print('visual_feat: ', visual_feat[0])
		# input()
		return q_values, belief_pred

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
			cross_entropy, belief_pred = self.train_belief_update(belief_update_batch, quick = True)
			# print('Teacher\'s belief estimate cross_entropy: %f' % cross_entropy)

			debug_structure['teacher_belief_prediction'] = belief_pred

			# for i in range(3):
			# 	print('\t target:', belief_update_batch['new_belief'][i, :],
			# 						np.max(belief_update_batch['new_belief'][i, :]))
			# 	print('\t predict', belief_pred[i, :], np.max(belief_pred[i, :]))

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
