import numpy as np
import tensorflow as tf

from pprint import pprint as brint
import pdb

from train import interact, restore_ckpt
from concept import Concept
from teacher import Teacher
from student import Student

from tqdm import tqdm

def main():
	attributes_size = 10
	num_distractors = 4
	concept_max_size = 4
	attributes = range(attributes_size)

	rl_gamma = 0.8
	boltzman_beta = 1
	belief_var_1d = 0
	epsilon = 0

	np.random.seed(425)

	concept_generator = Concept(attributes, num_distractors, concept_max_size)
	#test_batch = concept_generator.rd_generate_concept()
	#pdb.set_trace()

	sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, 
                                              log_device_placement = False))

	#in current design: msg_space_size == attribute_size
	with tf.device('/cpu:0'):
		teacher = Teacher(sess, rl_gamma, boltzman_beta, belief_var_1d, num_distractors, attributes_size, attributes_size)
		student = Student(sess, rl_gamma, boltzman_beta, belief_var_1d, num_distractors, attributes_size, attributes_size)

	init = tf.global_variables_initializer()
	sess.run(init)

	restore_ckpt(student, teacher, 'TOM_CL_phase4')
	accuracies = []
	rewards = []
	test_rounds = 1000
	hard_correct = [0, 0]
	all_wrongs = []
	for tr in tqdm(range(test_rounds)):
		trajectory_batch, belief_transit_batch, q_learning_batch, accuracy, reward, non_bayesian, debug_batch = \
			interact(teacher, student, concept_generator, epsilon, 0.9, 64, student_think = True)
		accuracies.append(accuracy)
		rewards.append(reward)
		long_batch = [db for db in debug_batch if len(db) > 1]
		g_long_batch = [lb for lb in long_batch if lb[0]['gain'] > 0]
		b_long_batch = [lb for lb in long_batch if lb[0]['gain'] < 0]
		bad_batch = [db for db in debug_batch if db[0]['gain'] < 0]
		nb_batch = [db for db in debug_batch if db[0]['non_bayesian'] > 0]
		hard_batch = [db for db in debug_batch if db[0]['teaching_dim'] > 1]
		g_hard_batch = [lb for lb in hard_batch if lb[0]['gain'] > 0]
		b_hard_batch = [lb for lb in hard_batch if lb[0]['gain'] < 0]
		hard_correct[0] += len(g_hard_batch)
		hard_correct[1] += len(hard_batch)
		all_wrongs.extend(bad_batch)
		# if len(nb_batch) > 0:#len(long_batch) > 0 or len(bad_batch) > 0:
		# 	pdb.set_trace()

	print(np.mean(accuracies), np.mean(rewards))
	print(1.0 * hard_correct[0] / hard_correct[1])
	pdb.set_trace()

if __name__ == '__main__':
	main()
