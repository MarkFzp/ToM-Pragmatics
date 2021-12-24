import os
import sys
import numpy as np
import tensorflow as tf
import argparse

from concept import Concept
from agent import Agent

from pprint import pprint as brint
import pdb

def interact(agentA, agentB, concept_generator,
			 epsilon, rl_gamma, batch_size, phase, limit_game_rd, ramp_cost, 
			 is_training = True, msg_penalty = 0.12):
	# belief_transit_batch = [(distractors, old_belief, msg, new_belief), ...]
	training_data_batch = {agentA.name_: {'q_learning_batch': [], 'trajectory_batch': [], 'belief_update_batch': []},
						   agentB.name_: {'q_learning_batch': [], 'trajectory_batch': [], 'belief_update_batch': []}}
	num_correct = 0
	total_reward = 0
	total_non_bayesian = 0
	debug_batch = []
	for b_idx in range(batch_size):
		calendar_A_idx, calendar_A_msgs, calendar_A_msg_tensor = concept_generator.rd_generate_concept(1)
		calendar_B_idx, calendar_B_msgs, calendar_B_msg_tensor = concept_generator.rd_generate_concept(1)
		msg_penalties = []
		
		#  A's calender and B's calender
		A_belief = concept_generator.tensor_[calendar_A_idx, :]
		B_belief = concept_generator.tensor_[calendar_B_idx, :]
		
		confirmation = concept_generator.num_slots_
		speak_turn = np.random.randint(2)
		speaker = agentA.name_ if speak_turn % 2 == 0 else agentB.name_
		listener = agentB.name_ if speak_turn % 2 == 0 else agentA.name_
		
		uniform_belief = 0.5 * np.ones(agentA.num_slots_)

		agent_mental_states = {agentA.name_:[], agentB.name_:[]}

		#agent, bss, bso, bos
		agent_mental_states[agentA.name_].extend([agentA, A_belief, uniform_belief, uniform_belief])
		agent_mental_states[agentB.name_].extend([agentB, B_belief, uniform_belief, uniform_belief])

		temp_trajectory = []

		joint_calendar = concept_generator.tensor_[calendar_A_idx, :] + concept_generator.tensor_[calendar_B_idx, :]
		meetable = int(np.sum(joint_calendar == 0) > 0)

		#convert slot self-belief to binary to get all the messages and message tensor
		bayesian_msgs = concept_generator.get_msg(int(''.join([str(e) for e in agent_mental_states[speaker][1]]), 2))
		msg_order = list(np.random.permutation(len(bayesian_msgs[0])))
		#action idx (confirmation): 
		# propose:[0 ~ num_slots-1]
		# wait: num_slots
		# reject: num_slots + 1
		debug_trajectory = []

		num_steps = 0
		#means while "hold"
		while confirmation == concept_generator.num_slots_:
			if limit_game_rd and num_steps == agentA.num_slots_:
				break
			
			if ramp_cost:
				msg_penalties.append(num_steps * msg_penalty)
			else:
				msg_penalties.append(msg_penalty)
			
			#simulate listener's belief update on all possible message and get their q value
			q_values, speaker_est_listener_belief = \
				agent_mental_states[speaker][0].get_q_value_for_all_msg(agent_mental_states[speaker][1],
																		agent_mental_states[speaker][2],
																		agent_mental_states[speaker][3],
																		concept_generator)
			rd = np.random.choice(2, 1, p = [1 - epsilon, epsilon])
			if phase == 0:
				msg_idx = msg_order.pop(0)
				msg_order.append(msg_idx)
				embed_msg = bayesian_msgs[1][msg_idx, :]
			else:	
				if rd == 1:
					msg_idx = np.random.randint(concept_generator.num_msgs_)
				else:
					maximums = np.argwhere(q_values == np.amax(q_values)).flatten().tolist()
					msg_idx = np.random.choice(maximums, 1)[0]

				embed_msg = concept_generator.all_msgs_tensor_[msg_idx, :]
			
			#lisener's updated belief on speaker, and the action it chose
			confirmation, listener_slots_belief, listener_belief_on_speaker, decision_prob, critic_value =\
				agent_mental_states[listener][0].update_self_belief_on_other(np.expand_dims(agent_mental_states[listener][2], 0),
																			 np.expand_dims(agent_mental_states[speaker][1], 0), 
																			 np.expand_dims(agent_mental_states[listener][1], 0),
																			 np.expand_dims(embed_msg, 0), concept_generator, is_training)

			#record listener's old belief on speaker and the action chosen
			temp_trajectory.append({'distractors': concept_generator.tensor_3d_,
									'self_belief_on_other': agent_mental_states[listener][2],
									'self_belief_on_self': agent_mental_states[listener][1],
									'message': embed_msg, 'action': confirmation,
									'critic_value': critic_value,
									'other_belief_on_other': agent_mental_states[speaker][1]})
			#record speaker's belief of other's belief on self, and the updated  other's belief on self
			training_data_batch[speaker]['belief_update_batch'].append((concept_generator.tensor_3d_,
																 	   agent_mental_states[speaker][3],
																	   embed_msg, listener_belief_on_speaker[0]))
			if phase != 0 and confirmation == concept_generator.num_slots_ and is_training:
				q_values_next, _ = agent_mental_states[listener][0].get_q_value_for_all_msg(agent_mental_states[listener][1],
																							   listener_belief_on_speaker[0],
																							   agent_mental_states[listener][3],
																							   concept_generator)
			else:
				q_values_next = [0]
			
			reward = 0
			if confirmation < concept_generator.num_slots_:
				reward = 1 if joint_calendar[confirmation] == 0 else -2
			elif confirmation == concept_generator.num_slots_ + 1: #listener claim cannot meet
				reward = 1 if not meetable else -2
			
			#if the number of rounds of game is limited and the last round is hold, give negative reward
			if limit_game_rd and reward == 0 and num_steps == agentA.num_slots_ - 1:
				reward = -2
			training_data_batch[speaker]['q_learning_batch'].append({'distractors': concept_generator.tensor_3d_,
															   		 'other_belief_on_self': agent_mental_states[speaker][3],
															   		 'self_belief_on_self': agent_mental_states[speaker][1],
															   		 'self_belief_on_other': agent_mental_states[speaker][2],
																	 'message': embed_msg,
															   	 	 'target_q': reward - msg_penalties[num_steps] +\
									 										   		 (confirmation == concept_generator.num_slots_) *\
																			   		 rl_gamma * max(q_values_next)})
			debug_trajectory.append({'speaker': speaker, 'listener': listener,
									 'listener_slot_belief': listener_slots_belief,
									 'listener_action_prob': decision_prob,
									 'listener_action': confirmation,
									 'meetable': meetable,
									 'joint_calendar': joint_calendar,
									 'speaker_calendar': agent_mental_states[speaker][1],
									 'listener_calendar': agent_mental_states[listener][1],
									 'speaker_est_listener_on_speaker': agent_mental_states[speaker][3],
									 'listener_on_speaker_prev': agent_mental_states[listener][2],
									 'listener_on_speaker_new': listener_belief_on_speaker[0],
									 'message': embed_msg, 'critic_value': critic_value,
									 })
			# update agent mental states
			#in training, directly put listener's updated "belief on other" to speaker's "other's belief on self"
			if is_training:
				agent_mental_states[speaker][3] = listener_belief_on_speaker[0]
			else:
				agent_mental_states[speaker][3] = speaker_est_listener_belief[0]
			
			#update listener's new belief on speaker
			agent_mental_states[listener][2] = listener_belief_on_speaker[0]
			
			#train policy as single agent in phase 0
			if phase != 0:
				speaker, listener = listener, speaker

			num_steps += 1

		for i in range(len(temp_trajectory)):
			temp_trajectory[i]['gain'] = (-1 * (np.power(rl_gamma, len(temp_trajectory) - i) - 1) / (rl_gamma - 1) * msg_penalties[i] +\
					np.power(rl_gamma, len(temp_trajectory) - i - 1) * reward)
			debug_trajectory[i]['gain'] = (-1 * (np.power(rl_gamma, len(temp_trajectory) - i) - 1) / (rl_gamma - 1) * msg_penalties[i] +\
					np.power(rl_gamma, len(temp_trajectory) - i - 1) * reward)


		training_data_batch[listener]['trajectory_batch'].append(temp_trajectory)
		debug_batch.append(debug_trajectory)

		total_reward -= sum(msg_penalties)
		# total_reward -= (num_steps - 1) * msg_penalty * len(temp_trajectory) / 2
		# total_reward -= len(temp_trajectory) * msg_penalty
		
		num_correct += (reward == 1)
		total_reward += reward

	accuracy = (1.0 * num_correct / (batch_size))

	return training_data_batch, accuracy, total_reward, debug_batch

def save_ckpt(agentA, agentB, ckpt_dir, ckpt_name):
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)
	if not os.path.exists(os.path.join(ckpt_dir, agentA.name_)):
		os.makedirs(os.path.join(ckpt_dir, agentA.name_))
	if not os.path.exists(os.path.join(ckpt_dir, agentB.name_)):
		os.makedirs(os.path.join(ckpt_dir, agentB.name_))

	agentA.total_saver_.save(agentA.sess, os.path.join(ckpt_dir, agentA.name_, ckpt_name))
	agentB.total_saver_.save(agentB.sess, os.path.join(ckpt_dir, agentB.name_, ckpt_name))
	print('Saved to %s' % ckpt_dir)

def restore_ckpt(agentA, agentB, ckpt_dir):
	ckpt_teacher = tf.train.get_checkpoint_state(os.path.join(ckpt_dir, agentA.name_))
	ckpt_student = tf.train.get_checkpoint_state(os.path.join(ckpt_dir, agentB.name_))
	if ckpt_teacher:
		agentA.total_loader_.restore(agentA.sess, ckpt_teacher.model_checkpoint_path)
	if ckpt_student:
		agentB.total_loader_.restore(agentB.sess, ckpt_student.model_checkpoint_path)
		pass
	if ckpt_student and ckpt_teacher:
		print('Load model from %s' % ckpt_dir)
		return True
	print('Fail to load model from %s' % ckpt_dir)
	return False



def main():
	
	#parsing arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-n','--num_slots', type=int, required = True)
	parser.add_argument('-l','--limit_step', action = 'store_true')
	parser.add_argument('-i','--init_msg_cost', type = float)
	args = parser.parse_args()
	
	#initilize arguments
	num_slots = args.num_slots
	limit_step = args.limit_step
	init_msg_cost = args.init_msg_cost
	ramp_cost = True if init_msg_cost is not None else False

	
	print("num_slot: {}, limit_step: {}, init_msg_cost: {}, ramp_cost: {}".format(num_slots, limit_step, init_msg_cost, ramp_cost))
	
	rl_gamma = 0.95

	calendars = Concept(num_slots)
	sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True,
	   										  log_device_placement = False))

	with tf.device('/gpu:0'):
		agentA = Agent('A', sess, num_slots)
		agentB = Agent('B', sess, num_slots)

	init = tf.global_variables_initializer()
	sess.run(init)
	#pretrain teacher for Bayesian belief udpate
	agent_pretraining_steps = 50000
	agent_pretrain_batch_size = 128
	agent_pretrain_ckpt_dir = 'Bayesian_Pretrain_%d_FCBU_exp' % num_slots
	agent_pretrain_ckpt_name = 'Belief_Update'

	agentA.pretrain_bayesian_belief_update(calendars, agent_pretraining_steps, agent_pretrain_batch_size,
											agent_pretrain_ckpt_dir, agent_pretrain_ckpt_name, continue_steps = 0)

	agentB.pretrain_bayesian_belief_update(calendars, agent_pretraining_steps, agent_pretrain_batch_size,
											agent_pretrain_ckpt_dir, agent_pretrain_ckpt_name, continue_steps = 0)

	rl_batch_size = 256
	continue_steps = 4e4
	phase = 0
	restored = restore_ckpt(agentA, agentB, 'TOM_CL_%d_slots_phase_%d_3000' % (num_slots, phase))
	if not restored  or (continue_steps > 0):
		communicative_learning_steps_0 = 5e4 if not restored else continue_steps
		global_step = 0
		accuracies = []
		rewards = []
		epsilon_min = 0.05
		epsilon_start = 0.95
		epsilon_decay = 1000
		while global_step < communicative_learning_steps_0:
			epsilon = 0#epsilon_min + (epsilon_start - epsilon_min) * np.exp (-1 * global_step / epsilon_decay)
			
			trajectory_batch, accuracy, reward, debug_batch = interact(agentA, agentB, calendars, epsilon, rl_gamma, rl_batch_size, phase, limit_game_rd = limit_step, ramp_cost = ramp_cost, msg_penalty = init_msg_cost)
			accuracies.append(accuracy)
			rewards.append(reward)
			agentA_debug = agentA.update_net(trajectory_batch[agentA.name_], phase)
			agentB_debug = agentB.update_net(trajectory_batch[agentB.name_], phase)
			ridx = np.random.randint(rl_batch_size)
			print('listener calender:', debug_batch[ridx][0]['listener_calendar'])
			print('speaker  calender:', debug_batch[ridx][0]['speaker_calendar'])
			print('joint    calendar:', debug_batch[ridx][0]['joint_calendar'])
			print('meetable:', debug_batch[ridx][0]['meetable'])
			print('message          :', debug_batch[ridx][0]['message'])
			print('listener_on_speaker_prev: ', debug_batch[ridx][0]['listener_on_speaker_prev'])
			print('listener_on_speaker_new: ', debug_batch[ridx][0]['listener_on_speaker_new'])
			print('#listener_slot_belief:', debug_batch[ridx][0]['listener_slot_belief'])
			print('$listener action Prob:', debug_batch[ridx][0]['listener_action_prob'])
			print('listener action: %d, return %f' % (debug_batch[ridx][0]['listener_action'], debug_batch[ridx][0]['gain']))
			if len(debug_batch[ridx]) > 1:
				print('@@@@@@@@@@@@@@@@')
				print('listener calender:', debug_batch[ridx][1]['listener_calendar'])
				print('speaker  calender:', debug_batch[ridx][1]['speaker_calendar'])
				print('message          :', debug_batch[ridx][1]['message'])
				print('listener_on_speaker_prev: ', debug_batch[ridx][1]['listener_on_speaker_prev'])
				print('listener_on_speaker_new: ', debug_batch[ridx][1]['listener_on_speaker_new'])
				print('#listener_slot_belief:', debug_batch[ridx][1]['listener_slot_belief'])
				print('$listener action Prob:', debug_batch[ridx][1]['listener_action_prob'])
				print('listener action: %d, return %f' % (debug_batch[ridx][1]['listener_action'], debug_batch[ridx][1]['gain']))
			print('[P%d][%d]Accuracy of this trajectory batch is %f, mean reward is %f' % (phase, global_step, accuracy, reward / rl_batch_size))
			global_step += 1
			if global_step % 100 == 0:
				#pdb.set_trace()
				pass
			if global_step % 3000 == 0:
				print('Mean accuracies: %f' % np.mean(accuracies))
				print('Mean rewards: %f' % np.mean(rewards))
				rewards = []
				accuracies = []
				save_ckpt(agentA, agentB, 'TOM_CL_%d_slots_phase_%d_%d' % (num_slots, phase, global_step), 'value')
	
	continue_steps = 0
	restored = restore_ckpt(student, teacher, 'TOM_CL_phase2')
	if not restored or (continue_steps > 0):
		communicative_learning_steps_2 = 8e4 if not restored else continue_steps
		accuracies = []
		rewards = []
		wrongs = []
		global_step = 0
		epsilon_min = 0.05
		epsilon_start = 0.95
		epsilon_decay = 2000
		while global_step < communicative_learning_steps_2:
			epsilon = 0#epsilon_min + (epsilon_start - epsilon_min) * np.exp (-1 * global_step / epsilon_decay)
			trajectory_batch, belief_transit_batch, q_learning_batch, accuracy, reward, non_bayesian, debug_batch = \
				interact(teacher, student, concept_generator, epsilon, rl_gamma, 64, student_think = False)
			accuracies.append(accuracy)
			rewards.append(reward)
			wrong = len([lb for lb in debug_batch if np.sum(lb[0]['student_new']) == 0])
			wrongs.append(wrong)
			print('[P2][%d]Accuracy of this trajectory batch is %f, total_reward is %f, num of wrong is %d, num of non-B: %d' \
					% (global_step, accuracy, reward, wrong, non_bayesian))
			if global_step <= 4e4:
				stu_loss_pg, stu_loss_vi = student.update_net(trajectory_batch, 3)
				print('\t stu pg loss: %f, stu vi: %f' % (stu_loss_pg, stu_loss_vi))
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

			if global_step % 10000 == 0:
				save_ckpt(student, teacher, 'TOM_CL_phase2_%d' % global_step, 'separate')

	continue_steps = 0
	restored = restore_ckpt(student, teacher, 'TOM_CL_phase3')
	if not restored or (continue_steps > 0):
		communicative_learning_steps_3 = 4e4 if not restored else continue_steps
		accuracies = []
		rewards = []
		wrongs = []
		global_step = 0
		epsilon_min = 0.05
		epsilon_start = 0.95
		epsilon_decay = 1000
		while global_step < communicative_learning_steps_3:
			epsilon = 0 if global_step >= 2e4 \
				else epsilon_min + (epsilon_start - epsilon_min) * np.exp (-1 * global_step / epsilon_decay)
			trajectory_batch, belief_transit_batch, q_learning_batch, accuracy, reward, non_bayesian, debug_batch = \
				interact(teacher, student, concept_generator, epsilon, rl_gamma, 64, student_think = False)
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
				interact(teacher, student, concept_generator, epsilon, rl_gamma, 64, student_think = False)
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