import os
import sys
import time
from math import pow
import argparse

import numpy as np
import tensorflow as tf
from pprint import pprint as brint

from Agents.teacher import Teacher
from Agents.student import Student

import pdb

def interact(teacher, student, concept_generator,
			 teacher_epsilon, rl_gamma, rational_beta,
			 batch_size, msg_penalty = 0.2,
			 student_think = False, student_wait = False):
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
		discrete_concepts, included_attributes, teacher_distractors, \
			student_distractors, t2s_msg, s2t_concpt, gt_idx = concept_generator.rd_generate_concept()
		

		# tea_cosine_max = tea_cosine[np.argsort(tea_cosine)[-2]]
		# stu_cosine_max = stu_cosine[np.argsort(stu_cosine)[-2]]
		# tea_cosine_min = tea_cosine[np.argsort(tea_cosine)[0]]
		# stu_cosine_min = stu_cosine[np.argsort(stu_cosine)[0]]
		# print(tea_cosine_mean - stu_cosine_mean)

		teacher_belief = np.zeros(teacher.num_distractors_)
		teacher_belief[gt_idx] = 1
		teacher_belief_stu = np.zeros(teacher.num_distractors_)
		teacher_belief_stu[np.where(s2t_concpt == gt_idx)[0][0]] = 1
		student_belief = np.ones(student.num_distractors_) / student.num_distractors_
		student_belief_tea = np.ones(student.num_distractors_) / student.num_distractors_

		# if student_think:
		# 	td_dict, _ = concept_generator.teaching_dim(discrete_concepts[1], included_attributes[1])
		# 	rtd = concept_generator.recursive_teaching_dim(discrete_concepts[1])
		# 	levels = concept_generator.teaching_level_for_each_concept(discrete_concepts[1])
		stu_prediction = teacher.num_distractors_ + 1
		
		while stu_prediction == teacher.num_distractors_ + 1:
			q_values, teacher_est_stu_belief = \
				teacher.get_q_value_for_all_msg(teacher_belief, student_belief_tea, teacher_distractors)
			rd = np.random.choice(2, 1, p = [1 - teacher_epsilon, teacher_epsilon])
			if rd == 1:
				msg_idx = np.random.randint(teacher.message_space_size_)
			elif teacher_epsilon == 0 and not student_think:
				expq = np.exp(rational_beta * q_values)
				msg_prob = expq / np.sum(expq)
				if not student_think:
					msg_prob_stu = -1 * np.ones(teacher.message_space_size_)
					for i in range(teacher.message_space_size_):
						msg_prob_stu[i] = msg_prob[np.where(t2s_msg == i)[0][0]]
				msg_idx = int(np.random.choice(teacher.message_space_size_, 1, p = msg_prob))
			else:
				maximums = np.argwhere(q_values == np.amax(q_values)).flatten().tolist()
				msg_idx = int(np.random.choice(maximums, 1))
			embed_msg_tea = np.zeros(teacher.message_space_size_)
			embed_msg_stu = np.zeros(teacher.message_space_size_)
			embed_msg_tea[msg_idx] = 1
			embed_msg_stu[t2s_msg[msg_idx]] = 1

			stu_prediction, stu_new_belief, stu_action_prob, critic_value, stu_msg_est =\
				student.get_prediction_new_belief(np.expand_dims(student_distractors, 0),
												  np.expand_dims(student_belief, 0),
												  np.expand_dims(embed_msg_stu, 0),
												  np.expand_dims(teacher_belief_stu, 0),
												  student_think, student_wait)
			stu_new_belief_tea = np.zeros(teacher.num_distractors_)
			for i in range(teacher.num_distractors_):
				# try:
				stu_new_belief_tea[i] = stu_new_belief[np.where(s2t_concpt == i)[0][0]]
				# except:
				# 	pdb.set_trace()
			bayesian_belief = concept_generator.bayesian_update(student_belief, discrete_concepts[1], t2s_msg[msg_idx])
			non_bayesian = np.sum((bayesian_belief == 0) * (stu_new_belief > 1e-2))
			if non_bayesian > 0:
				total_non_bayesian += 1
			
			reward = 1 * int(stu_prediction == np.where(s2t_concpt == gt_idx)[0][0])
			
			trajectory.append({'distractors': student_distractors,
							   'belief': student_belief,
							   'bayesian_belief': bayesian_belief,
							   'message': embed_msg_stu,
							   'action': stu_prediction,
							   'critic_value': critic_value,
							   'correct_answer': teacher_belief_stu})
			
			if student_think:
				tea_target = teacher_distractors[gt_idx]
				tea_cosine = np.sum(teacher_distractors * tea_target, axis = 1) / (np.sqrt(np.sum(teacher_distractors ** 2, axis = 1)) * np.sqrt(np.sum(tea_target ** 2)))
				stu_target = student_distractors[s2t_concpt.tolist().index(gt_idx)]
				stu_cosine = np.sum(student_distractors * stu_target, axis = 1) / (np.sqrt(np.sum(student_distractors ** 2, axis = 1)) * np.sqrt(np.sum(stu_target ** 2)))

				tea_cosine_mean = np.mean(tea_cosine[np.argsort(tea_cosine)[:-1]])
				stu_cosine_mean = np.mean(stu_cosine[np.argsort(stu_cosine)[:-1]])

				debug_trajectory.append({'concepts': discrete_concepts[1],
										'teacher': teacher_belief_stu,
										# 'student_prev': student_belief,
										'student_new': stu_new_belief,
										# 'student_action': stu_action_prob,
										'pi': embed_msg_stu,
										'stu_dis': student_distractors, 
										'message': t2s_msg[msg_idx],
										# 'prediction': stu_prediction,
										# 'critic_value': critic_value,
										# 'non_bayesian': non_bayesian,
										# 'msg_Q': [(t2s_msg[i], q_values[i]) for i in range(teacher.message_space_size_)],
										# 'msg_prob': msg_prob_stu if teacher_epsilon == 0 and not student_think else 'max',
										# 'teacher_estimate_stu_new': teacher_est_stu_belief,
										# 'stu_msg_est': stu_msg_est,
										# 'teaching_dim':
											# td_dict[tuple(discrete_concepts[1][np.where(s2t_concpt == gt_idx)[0][0]])][0] if student_think else None,
										# 'recursive_teaching_dim': rtd if student_think else None,
										# 'levels': levels if student_think else None,
										# 'level_of_target': levels[np.nonzero(teacher_belief_stu)] if student_think else None, 
										# 'tea_target_idx': gt_idx,
										'cosine_mean': (tea_cosine_mean + stu_cosine_mean) / 2,
										'reward': reward})
			
			else:
				debug_trajectory.append({'concepts': discrete_concepts[1],
										'teacher': teacher_belief_stu,
										# 'student_prev': student_belief,
										'student_new': stu_new_belief,
										# 'student_action': stu_action_prob,
										'pi': embed_msg_stu,
										'stu_dis': student_distractors, 
										'message': t2s_msg[msg_idx]})
										# 'prediction': stu_prediction,
										# 'critic_value': critic_value,
										# 'non_bayesian': non_bayesian,
										# 'msg_Q': [(t2s_msg[i], q_values[i]) for i in range(teacher.message_space_size_)],
										# 'msg_prob': msg_prob_stu if teacher_epsilon == 0 and not student_think else 'max',
										# 'teacher_estimate_stu_new': teacher_est_stu_belief,
										# 'stu_msg_est': stu_msg_est,
										# 'teaching_dim':
										# 	td_dict[tuple(discrete_concepts[1][np.where(s2t_concpt == gt_idx)[0][0]])][0] if student_think else None,
										# 'recursive_teaching_dim': rtd if student_think else None,
										# 'levels': levels if student_think else None,
										# 'level_of_target': levels[np.nonzero(teacher_belief_stu)] if student_think else None, 
										# 'tea_target_idx': gt_idx})


								
									#  'novel_test': {'msg': msg_idx, 'target': tuple(sorted(discrete_concepts[0][gt_idx]))}})
			
			belief_transit_batch.append((teacher_distractors, student_belief_tea, embed_msg_tea, stu_new_belief_tea))

			#reward = -1 if stu_prediction == -1 else int(stu_prediction == gt_idx)
			
			q_values_next, _ = teacher.get_q_value_for_all_msg(teacher_belief, stu_new_belief_tea, teacher_distractors)
			q_learning_batch.append((teacher_distractors, student_belief_tea, teacher_belief, embed_msg_tea,
									(1 * (reward) - 0) - msg_penalty +\
									(stu_prediction == teacher.num_distractors_ + 1) * rl_gamma * max(q_values_next)))

			student_belief = stu_new_belief
			student_belief_tea = stu_new_belief_tea

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
		if stu_prediction == np.where(s2t_concpt == gt_idx)[0][0]:
			num_correct += 1

	accuracy = (1.0 * num_correct / (batch_size))

	return trajectory_batch, belief_transit_batch, q_learning_batch, accuracy, total_reward, total_non_bayesian, debug_batch

def save_ckpt(student, teacher, ckpt_dir, ckpt_name, global_step):
	global_step = int(global_step)
	ckpt_dir = '%s_%d' % (ckpt_dir, global_step)
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)
	if not os.path.exists(os.path.join(ckpt_dir, 'teacher')):
		os.makedirs(os.path.join(ckpt_dir, 'teacher'))
	if not os.path.exists(os.path.join(ckpt_dir, 'student')):
		os.makedirs(os.path.join(ckpt_dir, 'student'))

	student.total_saver_.save(student.sess_,
							  os.path.join(ckpt_dir, 'student', ckpt_name),
							  global_step = global_step)
	teacher.total_saver_.save(teacher.sess_,
							  os.path.join(ckpt_dir, 'teacher', ckpt_name),
							  global_step = global_step)
	print('Saved to %s' % ckpt_dir)

def restore_ckpt(student, teacher, ckpt_dir):
	ckpt_teacher = tf.train.get_checkpoint_state(os.path.join(ckpt_dir, 'teacher'))
	ckpt_student = tf.train.get_checkpoint_state(os.path.join(ckpt_dir, 'student'))
	if ckpt_teacher:
		teacher.total_loader_.restore(teacher.sess_, ckpt_teacher.model_checkpoint_path)
	if ckpt_student:
		student.total_loader_.restore(student.sess_, ckpt_student.model_checkpoint_path)
		pass
	if ckpt_student and ckpt_teacher:
		print('Load model from %s' % ckpt_dir)
		return True
	print('Fail to load model from %s' % ckpt_dir)
	return False



def main():
	parser = argparse.ArgumentParser(description = 'train ToM agents')
	parser.add_argument('exp_folder', help = 'path to experiment folder')
	parser.add_argument('-g', type = int, required = True, help = 'gpu/cpu index, -1 is cpu, >= 0 is gpu index')
	# parser.add_argument('-r', type = int, help = 'index of test round')
	parser.add_argument('--pretrain', action = 'store_true', default = False, help = 'pretrain flag')
	
	args = parser.parse_args()
	
	exp_folder = args.exp_folder
	is_pretrain = args.pretrain
	processor_idx = args.g

	if is_pretrain:
		config_type = '_pretrain'
	else:
		config_type = ''


	if not os.path.isdir(os.path.join('./Experiments', exp_folder)):
		print('Cannot find target folder')
		exit()
	if (not os.path.exists(os.path.join('./Experiments', exp_folder, 'config.py')) and config_type == '')\
		or (not os.path.exists(os.path.join('./Experiments', exp_folder, 'config_pretrain.py'))
			and config_type == '_pretrain'):
		print('Cannot find config.py in target folder')
		exit()

	exec('from Experiments.%s.config%s import train_config, game_config, vision_config, belief_config, value_config'
		 % (exp_folder, config_type), globals())
	train_config.save_dir = os.path.join('./Experiments', exp_folder)
	game_config.save_dir = train_config.save_dir

	# if train_config.device.find('cpu') == 1:
	if processor_idx == -1:
		os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
	config.gpu_options.allow_growth = True
	sess = tf.Session(config = config)
	# sess = tf.Session()

	if processor_idx == -1:
		train_config.device = '/cpu:0'
	else:
		train_config.device = '/gpu:{}'.format(processor_idx)

	with tf.device(train_config.device):
		teacher = Teacher(sess, train_config, game_config, vision_config, belief_config, value_config)
		student = Student(sess, train_config, game_config, vision_config, belief_config, value_config)

	init = tf.global_variables_initializer()
	sess.run(init)

	exec('from %s.concept import Concept' % game_config.dataset, globals())
	concept_generator = Concept(game_config, is_train = True)

	teacher.pretrain_bayesian_belief_update(concept_generator)
	student.pretrain_bayesian_belief_update(concept_generator)

	if config_type == '_pretrain':
		print('Pretrain Finish')
		if vision_config.type != 'Feat':
			concept_generator.store_features(teacher, student)
			print('Finish saving features to experiment folder')
		exit()

	#np.random.seed(425)
	continue_steps = 0
	restored = restore_ckpt(student, teacher, os.path.join(train_config.save_dir, 'TOM_CL_phase1_20000'))
	if not restored  or (continue_steps > 0):
		if restored:
			train_config.iteration_phase1 = continue_steps
		global_step = 0
		accuracies = []
		rewards = []
		epsilon_min = 0.05
		epsilon_start = 0.95
		epsilon_decay = train_config.iteration_phase1 / 5
		while global_step <= train_config.iteration_phase1:
			epsilon = epsilon_min + (epsilon_start - epsilon_min) * np.exp (-1 * global_step / epsilon_decay)
			trajectory_batch, belief_transit_batch, q_learning_batch, accuracy, reward, non_bayesian, debug_batch = \
				interact(teacher, student, concept_generator, epsilon, game_config.rl_gamma,
						 game_config.rational_beta, train_config.batch_size)
			accuracies.append(accuracy)
			rewards.append(reward)
			teacher_debug = teacher.update_net(belief_transit_batch, q_learning_batch, 'Both')
			wrong_batch = [lb for lb in debug_batch if np.sum(lb[0]['student_new'][0: student.num_distractors_]) < 0.1]
			wrong = len(wrong_batch)
			# print('[P1][%d]Accuracy of this trajectory batch is %f, epsilon: %f, total_reward is %f, num of wrong is %d, num of non-B: %d' \
			# 		% (global_step, accuracy, epsilon, reward, wrong, non_bayesian))
			# if accuracy >= 0:
			# 	stu_loss_pg = student.update_net(trajectory_batch, 0)
			# 	print('\t stu pg: %f' % stu_loss_pg)
			global_step += 1
			if global_step % 500 == 0:
				print('[P1/%d]' % global_step)
				print('Mean accuracies: %f' % np.mean(accuracies))
				print('Mean rewards: %f' % np.mean(rewards))
				rewards = []
				accuracies = []
			if global_step % 5000 == 0:
				save_ckpt(student, teacher, os.path.join(train_config.save_dir, 'TOM_CL_phase1'), 'value', global_step)


	continue_steps = 10000
	restored = restore_ckpt(student, teacher, os.path.join(train_config.save_dir, 'TOM_CL_phase2_30000'))
	if not restored or (continue_steps > 0):
		if restored:
			train_config.iteration_phase2 = continue_steps
		accuracies = []
		rewards = []
		wrongs = []
		global_step = 0
		epsilon_min = 0.05
		epsilon_start = 0.95
		epsilon_decay = 5000
		while global_step < train_config.iteration_phase2:
			epsilon = 0 if global_step <= train_config.iteration_phase2 / 2\
						else epsilon_min + (epsilon_start - epsilon_min) * np.exp (-1 * (global_step - train_config.iteration_phase2 / 2) / epsilon_decay)
			trajectory_batch, belief_transit_batch, q_learning_batch, accuracy, reward, non_bayesian, debug_batch = \
				interact(teacher, student, concept_generator, epsilon, game_config.rl_gamma,
						 game_config.rational_beta, train_config.batch_size)
			accuracies.append(accuracy)
			rewards.append(reward)
			wrong_batch = [lb for lb in debug_batch if np.sum(lb[0]['student_new'][0: student.num_distractors_]) < 0.1]
			wrong = len(wrong_batch)
			wrongs.append(wrong)
			# print('[P2][%d]Accuracy of this trajectory batch is %f, total_reward is %f, num of wrong is %d, num of non-B: %d' \
			# 		% (global_step, accuracy, reward, wrong, non_bayesian))
			#pdb.set_trace()
			if global_step <= train_config.iteration_phase2 / 2:
				stu_loss_pg = student.update_net(trajectory_batch, 1)
				#print('\t stu pg loss: %f' % (stu_loss_pg))
			else:
				teacher_debug = teacher.update_net(belief_transit_batch, q_learning_batch, 'Both')
			global_step += 1
			if global_step % 200 == 0:
				print('[P2/%d]' % global_step)
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
				save_ckpt(student, teacher, os.path.join(train_config.save_dir, 'TOM_CL_phase2'), 'separate', global_step)

	continue_steps = 0
	restored = restore_ckpt(student, teacher, os.path.join(train_config.save_dir, 'TOM_CL_phase3'))
	if not restored or (continue_steps > 0):
		if restored:
			train_config.iteration_phase3 = continue_steps
		accuracies = []
		rewards = []
		wrongs = []
		cheats = []
		global_step = 0
		epsilon_min = 0.05
		epsilon_start = 0.95
		epsilon_decay = 1000
		while global_step < train_config.iteration_phase3:
			epsilon = 0 #if global_step >= 1e4 \
			# 	else epsilon_min + (epsilon_start - epsilon_min) * np.exp (-1 * global_step / epsilon_decay)
			trajectory_batch, belief_transit_batch, q_learning_batch, accuracy, reward, non_bayesian, debug_batch = \
				interact(teacher, student, concept_generator, epsilon, game_config.rl_gamma,
						 game_config.rational_beta, train_config.batch_size)
			accuracies.append(accuracy)
			rewards.append(reward)
			wrong = len([lb for lb in debug_batch if np.sum(lb[0]['student_new']) == 0])
			cheat = len([lb for lb in debug_batch if lb[0]['message'] not in lb[0]['concepts'][int(np.nonzero(lb[0]['teacher'])[0])]])
			wrongs.append(wrong)
			cheats.append(cheat)
			# print('[P3][%d]Accuracy of this trajectory batch is %f, total_reward is %f, num of wrong is %d, num of non-B: %d, num of screte: %d' \
			# 		% (global_step, accuracy, reward, wrong, non_bayesian, cheat))
			if global_step <= 1 * train_config.iteration_phase3 / 2:
				stu_loss_pg = student.update_net(trajectory_batch, 1)
				#print('\t stu pg loss: %f' % stu_loss_pg)
			else:
				teacher_debug = teacher.update_net(belief_transit_batch, q_learning_batch, 'Belief')
			global_step += 1
			if global_step % 200 == 0:
				print('[P3/%d]' % global_step)
				print('Mean accuracies: %f' % np.mean(accuracies))
				print('Mean rewards: %f' % np.mean(rewards))
				print('Mean wrongs: %f' % np.mean(wrongs))
				print('Mean cheats: %f' % np.mean(cheats))
				long_batch = [db for db in debug_batch if len(db) > 1]
				g_long_batch = [lb for lb in long_batch if lb[0]['gain'] > 0]
				b_long_batch = [lb for lb in long_batch if lb[0]['gain'] < 0]
				bad_batch = [db for db in debug_batch if db[0]['gain'] < 0]
				accuracies = []
				rewards = []
				wrongs = []
			if global_step % 5000 == 0:
				save_ckpt(student, teacher, os.path.join(train_config.save_dir, 'TOM_CL_phase3'), 'separate', global_step)

	print('Done phase 1-3')
	exit()
	continue_steps = 0
	restored = restore_ckpt(student, teacher, os.path.join(train_config.save_dir, 'TOM_CL_phase4'))
	if not restored or (continue_steps > 0):
		train_config.iteration_phase4 = 4e4 if not restored else continue_steps
		accuracies = []
		rewards = []
		wrongs = []
		global_step = 0
		while global_step < train_config.iteration_phase3:
			epsilon = 0
			trajectory_batch, belief_transit_batch, q_learning_batch, accuracy, reward, non_bayesian, debug_batch = \
				interact(teacher, student, concept_generator, epsilon, game_config.rl_gamma, game_config.rational_beta, train_config.batch_size, student_wait = True)
			accuracies.append(accuracy)
			rewards.append(reward)
			wrong = len([lb for lb in debug_batch if np.sum(lb[0]['student_new']) == 0])
			wrongs.append(wrong)
			print('[P4][%d]Accuracy of this trajectory batch is %f, total_reward is %f, num of wrong is %d, num of non-B: %d' \
					% (global_step, accuracy, reward, wrong, non_bayesian))
			stu_loss_pg = student.update_net(trajectory_batch, 2)
			print('\t stu pg loss: %f' % stu_loss_pg)
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
				save_ckpt(student, teacher, os.path.join(train_config.save_dir, 'TOM_CL_phase4'), 'separate', global_step)


if __name__ == '__main__':
	main()
