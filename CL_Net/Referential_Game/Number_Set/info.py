import numpy as np
import scipy.stats as sp
from concept import Concept

def info_gain(prev_dist, new_dist):
	return sp.entropy(prev_dist) - sp.entropy(new_dist)


def main():
	attributes = range(10)
	num_concepts = 5
	concept_size = 4
	concept_space = Concept(attributes, num_concepts, concept_size)
	problem1 = [(1, 2, 3, 4), (3, 4, 5, 6), (2, 4, 5, 7), (2, 3, 5, 8), (2, 3, 4, 5)]
	init_belief = np.ones(num_concepts) / num_concepts

	for msg in [2, 3, 4, 5]:
		new_belief = concept_space.bayesian_update(init_belief, problem1, msg)
		print(info_gain(init_belief, new_belief))
		init_belief = new_belief
	print(info_gain(np.ones(num_concepts) / num_concepts, new_belief))

	print('%%%%%%%%%%%%%%%%%%%%%%')
	problem2 = [(0, 2, 3), (4, 7, 9), (4, 7), (0, 2, 4, 9)]
	init_belief = np.ones(4) / 4
	for msg in [7] * 8:
		new_belief = concept_space.bayesian_update(init_belief, problem2, msg)
		print(info_gain(init_belief, new_belief))
		init_belief = new_belief
	print(info_gain(np.ones(4) / 4, [0, 0, 1, 0]))

if __name__ == '__main__':
	main()
