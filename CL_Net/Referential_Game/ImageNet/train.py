import os
import numpy as np
import tensorflow as tf
from math import pow

from concept import Concept
from teacher import Teacher
from student import Student

from pprint import pprint as brint
import pdb

def interact(teacher, student, concept_generator,
			 teacher_epsilon, rl_gamma, batch_size,
			 msg_penalty = 0.2, student_think = False, student_wait = False):
	# trajectory_batch = [[(distractors, belief, msg, action, advantage), ...], [], []]
	# belief_transit_batch = [(distractors, old_belief, msg, new_belief), ...]
	# q_learning_batch = [(distractors, student_belief, teacher_belief, msg, reward + Q), ...]
	trajectory_batch = []
	belief_transit_batch = []
	q_learning_batch = []
	debug_batch = []
	num_correct = 0
	total_reward = 0
	total_non_bayesian = 0
	for b_idx in range(batch_size):
		trajectory = []
		debug_trajectory = []
		discrete_concepts, _, included_attributes, images_tensor = concept_generator.rd_generate_concept()
		teacher_belief = np.zeros(teacher.num_distractors_)
		gt_idx = np.random.randint(teacher.num_distractors_)
		teacher_belief[gt_idx] = 1
		student_belief = np.ones(student.num_distractors_) / student.num_distractors_
		# td_dict, _ = concept_generator.teaching_dim(discrete_concepts, included_attributes)
		
		stu_prediction = teacher.num_distractors_ + 1
		while stu_prediction == teacher.num_distractors_ + 1:
			q_values, teacher_est_stu_belief, dist_feat, belief_dst, msg_est_tensor = \
				teacher.get_q_value_for_all_msg(teacher_belief, student_belief, images_tensor)
			rd = np.random.choice(2, 1, p = [1 - teacher_epsilon, teacher_epsilon])
			if rd == 1:
				msg_idx = np.random.randint(teacher.message_space_size_)
			else:
				maximums = np.argwhere(q_values == np.amax(q_values)).flatten().tolist()
				msg_idx = np.random.choice(maximums, 1)

			embed_msg = np.zeros(teacher.message_space_size_)
			embed_msg[msg_idx] = 1

			stu_prediction, stu_new_belief, stu_action_prob, critic_value, stu_msg_est =\
				student.get_prediction_new_belief(np.expand_dims(images_tensor, 0),
												  np.expand_dims(student_belief, 0),
												  np.expand_dims(embed_msg, 0),
												  np.expand_dims(teacher_belief, 0),
												  student_think, student_wait)

			# if concept_generator.teaching_dim(discrete_concepts, included_attributes)[0][discrete_concepts[gt_idx]][0] == 1:
			# 	critic_value = 0.8
			# else:
			# 	critic_value = 0
			bayesian_belief = concept_generator.bayesian_update(np.ones(teacher.num_distractors_) / teacher.num_distractors_, discrete_concepts, msg_idx)
			non_bayesian = np.sum((bayesian_belief == 0) * (stu_new_belief > 1e-2))
			if non_bayesian > 0:
				total_non_bayesian += 1
				#pdb.set_trace()
			
			trajectory.append({'distractors': images_tensor,
							   'belief': student_belief,
							   'bayesian_belief': bayesian_belief,
							   'message': embed_msg,
							   'action': stu_prediction,
							   'critic_value': critic_value,
							   'correct_answer': teacher_belief})
			
			#pdb.set_trace()
			debug_trajectory.append({'concepts': discrete_concepts,
									 'teacher': teacher_belief,
									 'student_prev': student_belief,
									 'student_new': stu_new_belief,
									 'student_action': stu_action_prob,
									 'message': msg_idx,
									 'prediction': stu_prediction,
									 'critic_value': critic_value,
									 'non_bayesian': non_bayesian,
									 'msg_Q': [(i, q_values[i]) for i in range(teacher.message_space_size_)],
									 'teacher_estimate_stu_new': teacher_est_stu_belief,
									 'dist_feat': dist_feat[0, ...],
									 'belief_dst': belief_dst,
									 'stu_msg_est': stu_msg_est})
									#  'teaching_dim': td_dict[tuple(discrete_concepts[gt_idx])][0]})
			
			belief_transit_batch.append((images_tensor, student_belief, embed_msg, stu_new_belief[0]))

			#reward = -1 if stu_prediction == -1 else int(stu_prediction == gt_idx)
			reward = int(stu_prediction == gt_idx)
			
			q_values_next, _, _, _, _ = teacher.get_q_value_for_all_msg(teacher_belief, stu_new_belief[0], images_tensor)
			q_learning_batch.append((images_tensor, student_belief, teacher_belief, embed_msg,
									(1 * (reward) - 0) - msg_penalty +\
									(stu_prediction == teacher.num_distractors_ + 1) * rl_gamma * max(q_values_next)))

			student_belief = stu_new_belief[0]

				# if len(trajectory) > 5:
				# 	pdb.set_trace()

		for i in range(len(trajectory)):
			if rl_gamma == 1:
				gain = pow(rl_gamma, len(trajectory) - i - 1) * (1 * reward - 0) - pow(msg_penalty, len(trajectory) - i)
			else:
				gain = pow(rl_gamma, len(trajectory) - i - 1) * (1 * reward - 0) - 1 * (pow(rl_gamma, len(trajectory) - i) - 1) / (rl_gamma - 1) * msg_penalty
			trajectory[i]['gain'] = gain
			debug_trajectory[i]['gain'] = gain

		trajectory_batch.append(trajectory)
		debug_batch.append(debug_trajectory)
		total_reward -= len(trajectory) * msg_penalty
		total_reward += reward
		if stu_prediction == gt_idx:
			num_correct += 1

	accuracy = (1.0 * num_correct / (batch_size))

	return trajectory_batch, belief_transit_batch, q_learning_batch, accuracy, total_reward, total_non_bayesian, debug_batch

def save_ckpt(student, teacher, ckpt_dir, ckpt_name):
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)
	if not os.path.exists(os.path.join(ckpt_dir, 'teacher')):
		os.makedirs(os.path.join(ckpt_dir, 'teacher'))
	if not os.path.exists(os.path.join(ckpt_dir, 'student')):
		os.makedirs(os.path.join(ckpt_dir, 'student'))

	student.total_saver_.save(student.sess, os.path.join(ckpt_dir, 'student', ckpt_name))
	teacher.total_saver_.save(teacher.sess, os.path.join(ckpt_dir, 'teacher', ckpt_name))
	print('Saved to %s' % ckpt_dir)

def restore_ckpt(student, teacher, ckpt_dir):
	ckpt_teacher = tf.train.get_checkpoint_state(os.path.join(ckpt_dir, 'teacher'))
	ckpt_student = tf.train.get_checkpoint_state(os.path.join(ckpt_dir, 'student'))
	if ckpt_teacher:
		teacher.total_loader_.restore(teacher.sess, ckpt_teacher.model_checkpoint_path)
	if ckpt_student:
		student.total_loader_.restore(student.sess, ckpt_student.model_checkpoint_path)
		pass
	if ckpt_student and ckpt_teacher:
		print('Load model from %s' % ckpt_dir)
		return True
	print('Fail to load model from %s' % ckpt_dir)
	return False



def main():
	attributes_size = 18
	num_distractors = 7
	img_length = 224
	attributes = range(18)
	n_grid_per_side = 2

	rl_gamma = 0.6
	boltzman_beta = 1
	belief_var_1d = -10


	sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, 
                                              log_device_placement = False))
	# from tensorflow.python import debug as tf_debug
	# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
	# sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

	# in current design: msg_space_size == attribute_size
	with tf.device("/device:GPU:0"):
		teacher = Teacher(sess, rl_gamma, boltzman_beta, belief_var_1d, num_distractors, attributes_size, attributes_size, img_length)
		student = Student(sess, rl_gamma, boltzman_beta, belief_var_1d, num_distractors, attributes_size, attributes_size, img_length)

	init = tf.global_variables_initializer()
	sess.run(init)
	#pretrain teacher for Bayesian belief udpate
	teacher_pretraining_steps = 2
	teacher_pretrain_batch_size = 20
	teacher_pretrain_ckpt_dir = 'Bayesian_Pretrain_Teacher_%f' % (belief_var_1d)
	teacher_pretrain_ckpt_name = 'Belief_Update'

	concept_generator = Concept(num_distractors, img_length=img_length, n_grid_per_side=n_grid_per_side)

	teacher.pretrain_bayesian_belief_update(concept_generator, teacher_pretraining_steps, teacher_pretrain_batch_size,
											teacher_pretrain_ckpt_dir, teacher_pretrain_ckpt_name, continue_steps = 0)

	student_pretraining_steps = 2
	student_pretrain_batch_size = 20
	student_pretrain_ckpt_dir = 'Bayesian_Pretrain_Student_%f' % (belief_var_1d)
	student_pretrain_ckpt_name = 'Belief_Update'

	student.pretrain_bayesian_belief_update(concept_generator, student_pretraining_steps, student_pretrain_batch_size,
											student_pretrain_ckpt_dir, student_pretrain_ckpt_name, continue_steps = 0)
	
	batch_size = 20

	continue_steps = 0
	restored = restore_ckpt(student, teacher, 'TOM_CL_phase1_25000')
	if not restored  or (continue_steps > 0):
		communicative_learning_steps_1 = 4e4 if not restored else continue_steps
		global_step = 0
		accuracies = []
		rewards = []
		epsilon_min = 0.05
		epsilon_start = 0.95
		epsilon_decay = 3000
		while global_step <= communicative_learning_steps_1:
			epsilon = epsilon_min + (epsilon_start - epsilon_min) * np.exp (-1 * global_step / epsilon_decay)
			trajectory_batch, belief_transit_batch, q_learning_batch, accuracy, reward, non_bayesian, debug_batch = \
				interact(teacher, student, concept_generator, epsilon, rl_gamma, batch_size)
			accuracies.append(accuracy)
			rewards.append(reward)
			teacher_debug = teacher.update_net(belief_transit_batch, q_learning_batch, 'Q-Net')
			wrong_batch = [lb for lb in debug_batch if np.sum(lb[0]['student_new'][0: student.num_distractors_]) < 0.1]
			wrong = len(wrong_batch)
			print('[P1][%d]Accuracy of this trajectory batch is %f, total_reward is %f, num of wrong is %d, num of non-B: %d' \
					% (global_step, accuracy, reward, wrong, non_bayesian))
			if accuracy >= 0:
				stu_loss_pg, stu_loss_vi = student.update_net(trajectory_batch, 2)
				print('\t stu pg: %f, stu vi: %f' % (stu_loss_pg, stu_loss_vi))

			global_step += 1
			if global_step % 5000 == 0:
				print('Mean accuracies: %f' % np.mean(accuracies))
				print('Mean rewards: %f' % np.mean(rewards))
				rewards = []
				accuracies = []

				save_ckpt(student, teacher, 'TOM_CL_phase1_%d' % global_step, 'value')
	
	# input('phase 1 finished')


	continue_steps = 7000
	restored = restore_ckpt(student, teacher, 'TOM_CL_phase2_5000')
	if not restored or (continue_steps > 0):
		communicative_learning_steps_2 = 4e4 if not restored else continue_steps
		accuracies = []
		rewards = []
		wrongs = []
		global_step = 5000
		epsilon_min = 0.05
		epsilon_start = 0.95
		epsilon_decay = 2000
		while global_step < communicative_learning_steps_2:
			epsilon = 0 #epsilon_min + (epsilon_start - epsilon_min) * np.exp (-1 * global_step / epsilon_decay)
			trajectory_batch, belief_transit_batch, q_learning_batch, accuracy, reward, non_bayesian, debug_batch = \
				interact(teacher, student, concept_generator, epsilon, rl_gamma, batch_size)
			accuracies.append(accuracy)
			rewards.append(reward)
			wrong = len([lb for lb in debug_batch if np.sum(lb[0]['student_new'][0: student.num_distractors_]) < 0.1])
			wrongs.append(wrong)
			print('[P2][%d]Accuracy of this trajectory batch is %f, total_reward is %f, num of wrong is %d, num of non-B: %d' \
					% (global_step, accuracy, reward, wrong, non_bayesian))
			#pdb.set_trace()
			if global_step <= 10e4:
				stu_loss_pg, _ = student.update_net(trajectory_batch, 3)
				# print('\t stu pg loss: %f, stu vi: %f' % (stu_loss_pg, stu_loss_vi))
				print('\t stu pg loss: %f' % (stu_loss_pg))
			else:
				teacher_debug = teacher.update_net(belief_transit_batch, q_learning_batch, 'Belief')
			global_step += 1
			if global_step % 200 == 0:
				print('Mean accuracies: %f' % np.mean(accuracies))
				print('Mean rewards: %f' % np.mean(rewards))
				print('Mean wrongs: %f' % np.mean(wrongs))
				long_batch = [db for db in debug_batch if len(db) > 1]
				g_long_batch = [lb for lb in long_batch if lb[0]['gain'] > 0]
				b_long_batch = [lb for lb in long_batch if lb[0]['gain'] < 0]
				bad_batch = [db for db in debug_batch if db[0]['gain'] < 0]
				accuracies = []
				rewards = []
				wrongs = []

			if global_step % 200 == 0:
				save_ckpt(student, teacher, 'TOM_CL_phase2_%d' % global_step, 'separate')

	continue_steps = 0
	restored = restore_ckpt(student, teacher, 'TOM_CL_phase3')
	if not restored or (continue_steps > 0):
		communicative_learning_steps_3 = 2e4 if not restored else continue_steps
		accuracies = []
		rewards = []
		wrongs = []
		global_step = 0
		epsilon_min = 0.05
		epsilon_start = 0.95
		epsilon_decay = 1000
		while global_step < communicative_learning_steps_3:
			epsilon = 0 if global_step >= 1e4 \
				else epsilon_min + (epsilon_start - epsilon_min) * np.exp (-1 * global_step / epsilon_decay)
			trajectory_batch, belief_transit_batch, q_learning_batch, accuracy, reward, non_bayesian, debug_batch = \
				interact(teacher, student, concept_generator, epsilon, rl_gamma, batch_size)
			accuracies.append(accuracy)
			rewards.append(reward)
			wrong = len([lb for lb in debug_batch if np.sum(lb[0]['student_new']) == 0])
			wrongs.append(wrong)
			print('[P3][%d]Accuracy of this trajectory batch is %f, total_reward is %f, num of wrong is %d, num of non-B: %d' \
					% (global_step, accuracy, reward, wrong, non_bayesian))
			if global_step >= 2e4:
				stu_loss_pg, stu_loss_vi = student.update_net(trajectory_batch, 3)
				print('\t stu pg loss: %f, stu vi: %f' % (stu_loss_pg, stu_loss_vi))
			else:
				teacher_debug = teacher.update_net(belief_transit_batch, q_learning_batch, 'Q-Net')
			global_step += 1
			if global_step % 200 == 0:
				print('Mean accuracies: %f' % np.mean(accuracies))
				print('Mean rewards: %f' % np.mean(rewards))
				print('Mean wrongs: %f' % np.mean(wrongs))
				long_batch = [db for db in debug_batch if len(db) > 1]
				g_long_batch = [lb for lb in long_batch if lb[0]['gain'] > 0]
				b_long_batch = [lb for lb in long_batch if lb[0]['gain'] < 0]
				bad_batch = [db for db in debug_batch if db[0]['gain'] < 0]
				accuracies = []
				rewards = []
				wrongs = []
			if global_step % 5000 == 0:
				save_ckpt(student, teacher, 'TOM_CL_phase3_%d' % global_step, 'separate')


	continue_steps = 0
	restored = restore_ckpt(student, teacher, 'TOM_CL_phase4')
	if not restored or (continue_steps > 0):
		communicative_learning_steps_4 = 4e4 if not restored else continue_steps
		accuracies = []
		rewards = []
		wrongs = []
		global_step = 0
		while global_step < communicative_learning_steps_3:
			epsilon = 0
			trajectory_batch, belief_transit_batch, q_learning_batch, accuracy, reward, non_bayesian, debug_batch = \
				interact(teacher, student, concept_generator, epsilon, rl_gamma, batch_size, student_wait = True)
			accuracies.append(accuracy)
			rewards.append(reward)
			wrong = len([lb for lb in debug_batch if np.sum(lb[0]['student_new']) == 0])
			wrongs.append(wrong)
			print('[P4][%d]Accuracy of this trajectory batch is %f, total_reward is %f, num of wrong is %d, num of non-B: %d' \
					% (global_step, accuracy, reward, wrong, non_bayesian))
			stu_loss_pg, stu_loss_vi = student.update_net(trajectory_batch, 1)
			print('\t stu pg loss: %f, stu vi: %f' % (stu_loss_pg, stu_loss_vi))
			global_step += 1
			if global_step % 200 == 0:
				print('Mean accuracies: %f' % np.mean(accuracies))
				print('Mean rewards: %f' % np.mean(rewards))
				print('Mean wrongs: %f' % np.mean(wrongs))
				long_batch = [db for db in debug_batch if len(db) > 1]
				g_long_batch = [lb for lb in long_batch if lb[0]['gain'] > 0]
				b_long_batch = [lb for lb in long_batch if lb[0]['gain'] < 0]
				bad_batch = [db for db in debug_batch if db[0]['gain'] < 0]
				accuracies = []
				rewards = []
				wrongs = []

			if global_step % 10000 == 0:
				save_ckpt(student, teacher, 'TOM_CL_phase4_%d' % global_step, 'separate')


if __name__ == '__main__':
	main()
