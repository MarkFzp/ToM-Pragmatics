import numpy as np
from collections import defaultdict
import random
import os


'''
code book:
circle: 0
rect: 1
tri: 2
ellipse: 3
star: 4
loop: 5
red: 6
green: 7
blue: 8
yellow: 9
cyan: 10
magenta: 11
large: 12
small: 13
upper_left: 14
upper_right: 15
lower_left: 16
lower_right: 17
'''


def get_combination(samples, num):
	if num == 0:
		return [[]]
	else:
		combinations = []
		while len(samples) > 0:
			s = samples[0]
			samples = samples[1:]
			sub_combinations = get_combination(samples, num - 1)
			combinations.extend([sc + [s] for sc in sub_combinations])
		return combinations


class Concept:
	def __init__(self, game_config):
		self.attributes = game_config.attributes
		self.num_distractors = game_config.num_distractors
		self.concept_size = 4  # color, shape, size, position
		self.dataset_attributes_path = game_config.dataset_attributes_path
		self.img_h = game_config.img_h
		self.img_w = game_config.img_w
		self.save_dir = game_config.save_dir
		if not os.path.exists(self.dataset_attributes_path):
			raise Exception('dataset_attributes_path does not exist')
		self.dataset_attributes = np.load(self.dataset_attributes_path)
		assert(self.dataset_attributes.shape[1] == len(self.attributes))
		self.data_count = self.dataset_attributes.shape[0]
		if 'images_path' in game_config and game_config.images_path is not None:
			self.img_as_data = True
			self.images_path = game_config.images_path
			if not os.path.exists(self.images_path):
				raise Exception('images_path does not exist')
			self.images = np.load(self.images_path)
			assert(self.images.shape[1] == self.img_h and self.images.shape[2] == self.img_w)
			assert(self.images.shape[0] == self.data_count)
		else:
			if 'feat_dir' in game_config and game_config.feat_dir is not None:
				load_dir = game_config.feat_dir
			else:
				load_dir = self.save_dir
			self.img_as_data = False
			self.teacher_feat_path = os.path.join(load_dir, 'Geometry_Teacher_features.npy')
			self.student_feat_path = os.path.join(load_dir, 'Geometry_Student_features.npy')
			if not os.path.exists(self.teacher_feat_path) or not os.path.exists(self.student_feat_path):
				raise Exception('teacher_feat_path or student_feat_path does not exist')
			self.teacher_features = np.load(self.teacher_feat_path)
			self.student_features = np.load(self.student_feat_path)
			assert(self.teacher_features.shape == self.student_features.shape)
			assert(self.teacher_features.shape[0] == self.data_count)


	def store_features(self, teacher, student):
		def helper(agent):
			save_path = os.path.join(self.save_dir, 'Geometry_%s_features.npy' % agent.role_)
			mod = self.images.shape[0] % self.num_distractors
			if mod == 0:
				cnn_input = self.images.reshape([-1, self.num_distractors, *self.images.shape[1:]])
			else:
				dim_to_append = self.num_distractors - mod
				padding = np.zeros([dim_to_append, *self.images.shape[1:]])
				cnn_input = np.concatenate([self.images, padding], axis = 0).reshape([-1, self.num_distractors, *self.images.shape[1:]])
			cnn_output = agent.sess_.run(agent.perception_core_.visual_features_, feed_dict = {agent.distractors_: cnn_input})
			if mod == 0:
				features = cnn_output.reshape([self.images.shape[0], cnn_output.shape[-1]])
			else:
				features = cnn_output.reshape([self.images.shape[0] + dim_to_append, cnn_output.shape[-1]])[:self.images.shape[0]]
			np.save(save_path, features)
		helper(teacher)
		helper(student)


	def rd_generate_concept(self):
		chosen_idx = np.random.randint(0, self.data_count, size = self.num_distractors)
		concept_embed = self.dataset_attributes[chosen_idx]
		concepts = []
		included_attributes = set()
		for embed in concept_embed:
			embed_attributes = np.where(embed)[0].tolist()
			concepts.append(embed_attributes)
			included_attributes.update(embed_attributes)
		if self.img_as_data:
			distractors = self.images[chosen_idx]
			return concepts, list(included_attributes), distractors, distractors
		else:
			teacher_distractors = self.teacher_features[chosen_idx]
			student_distractors = self.student_features[chosen_idx]
			return concepts, list(included_attributes), teacher_distractors, student_distractors


	def teaching_dim(self, concepts, included_attrs):
		td_dict = {}
		teaching_sample = defaultdict(list)
		sample_size = 1
		smallest_sample_size = self.concept_size
		for i in range(len(concepts)):
			for j in range(len(concepts)):
				if set(concepts[i]).issubset(set(concepts[j])) and i != j:
					td_dict[tuple(concepts[i])] = (self.concept_size, tuple(concepts[i]))
		while len(td_dict) < len(concepts):
			all_teaching_samples = get_combination(included_attrs, sample_size)
			for ts in all_teaching_samples:
				for concept in concepts:
					if set(ts).issubset(set(concept)):
						teaching_sample[tuple(ts)].append(concept)
			for ts in teaching_sample:
				if len(teaching_sample[ts]) == 1:
					concept = teaching_sample[ts][0]
					if td_dict.get(tuple(concept)) is None:
						td_dict[tuple(concept)] = (sample_size, ts)
						smallest_sample_size = min(smallest_sample_size, sample_size)
			###
			# if len(td_dict) == len(concepts):
			# 	return True
			# else:
			# 	return False
			###
			sample_size += 1

		###
		# return False
		###
		return td_dict, smallest_sample_size


	def recursive_teaching_dim(self, concepts, current_most = 0):
		if len(concepts) == 0:
			return current_most
		included_attributes = []
		for c in concepts:
			for e in c:
				included_attributes.append(e)
		included_attributes = list(set(included_attributes))
		td_dict, smallest_sample_size = self.teaching_dim(concepts, included_attributes)
		new_concepts = [c for c in concepts if td_dict[tuple(c)][0] > smallest_sample_size]
		return self.recursive_teaching_dim(new_concepts, max(smallest_sample_size, current_most))


	def bayesian_update(self, old_belief, concepts, info):
		likelihood = []
		for concept in concepts:
			prob = 1.0 * (info in concept) / len(concept)
			likelihood.append(prob)
		new_belief = old_belief * np.array(likelihood)
		new_belief /= np.sum(new_belief) + 1e-9
		return new_belief


	def generate_batch(self, batch_size, role, epsilon = 0.4):
		data = {'prev_belief': [None] * batch_size,
				'message': [None] * batch_size,
				'distractors': [None] * batch_size,
				'new_belief': [None] * batch_size}
		if self.img_as_data:
			distractors = self.images
		else:
			if role == 'Teacher':
				distractors = self.teacher_features
			elif role == 'Student':
				distractors = self.student_feat_path
			else:
				raise Exception('Wrong role passed in generate_batch')
		for i in range(batch_size):
			chosen_idx = np.random.randint(0, self.data_count, size = self.num_distractors)
			concept_embed = self.dataset_attributes[chosen_idx]
			concepts = []
			included_attributes = set()
			for embed in concept_embed:
				embed_attributes = np.where(embed)[0].tolist()
				concepts.append(embed_attributes)
				included_attributes.update(embed_attributes)
			included_attributes = list(included_attributes)
			prev_belief = np.random.random(self.num_distractors)
			prev_belief /= np.sum(prev_belief)
			rd = np.random.choice(2, 1, p = [1 - epsilon, epsilon])
			if rd == 1:
				msg = included_attributes[np.random.randint(len(included_attributes))]
			else:
				msg = np.random.randint(len(self.attributes))
			embeded_msg = np.zeros(len(self.attributes))
			embeded_msg[msg] = 1
			new_belief = self.bayesian_update(prev_belief, concepts, msg)
			data['prev_belief'][i] = prev_belief
			data['message'][i] = embeded_msg
			data['distractors'][i] = distractors[chosen_idx]
			data['new_belief'][i] = new_belief
		for j in data:
			data[j] = np.array(data[j])
		return data


if __name__ == '__main__':
	pass
	# np.set_printoptions(threshold=np.nan)
	# num_concept = 7
	# # concepts, concept_embed, included_attrs = concept_space.rd_generate_concept()
	# # print(concepts)
	# # tensor = concept_space.get_images_tensor(concepts)
	# # for img in concept_space.tensor2ims(tensor):
	# # 	img.show()
	# # 	input()
	
	# import time
	# t1 = time.time()
	# data = concept_space.generate_batch(1000)
	# diff = time.time() - t1
	# print(diff)

	# ims = concept_space.images_tensor_mean[[0, 4, 0, 4]]
	# ims_scaled = []
	# for im in ims:
	# 	im -= np.min(im)
	# 	im /= np.max(im)
	# 	im *= 255
	# 	ims_scaled.append(im)
	# ims_scaled = np.array(ims_scaled)
	# for im in concept_space.tensor2ims(ims_scaled):
	# 	im.show()
	# input()

	# print(concept_space.images_tensor.shape)
	# im = np.mean(concept_space.images_tensor, axis=0)
	# im -= np.min(im)
	# im /= np.max(im)
	# im *= 255
	# concept_space.arr2im(im).show()

	# print(data['prev_belief'][:10])
	# print(data['new_belief'][:10])
	# print((1000 - np.count_nonzero(np.sum(data['new_belief'], axis=1))) / 1000)

	# for im in concept_space.images_tensor_mean:
	# 	print(np.min(im))
	# 	im -= np.min(im)
	# 	print(np.min(im))
	# 	im /= np.max(im)
	# 	im *= 255
	# 	print(np.min(im))
	# 	# concept_space.arr2im(im).show()

	# for num_concept in range(4, 8):
	# 	concept_space = Concept(num_concept, n_grid_per_side=3)
	# 	count = 0
	# 	for i in range(100000):
	# 		concepts, _, included_attributes, _ = concept_space.rd_generate_concept()
	# 		if concept_space.teaching_dim(concepts, included_attributes):
	# 			count += 1
	# 	print("mujoco with {} dis and 9 grids: {}".format(num_concept, count / 100000))

	# for i in [-4, -8]:
	# 	im_arr = concept_space.images_tensor[i]
	# 	concept_space.arr2im(im_arr).show()

	
	
	# concept_space.arr2im(arr).show()
	# map = concept_space.get_att_map_arr(concepts, [1/num_concept for i in range(num_concept)])
	# print(map)
	# concept_space.arr2im(map).show()
	# target_map = concept_space.target_att_map_arr(concepts, 1)
	# concept_space.arr2im(target_map).show()

	# prior = np.ones(num_concept) / num_concept
	# belief1 = concept_space.bayesian_update(prior, concepts, concepts[1][0])
	# belief2 = concept_space.bayesian_update(belief1, concepts, concepts[1][1])
	# belief3 = concept_space.bayesian_update(belief2, concepts, concepts[1][2])
	# belief4 = concept_space.bayesian_update(belief2, concepts, concepts[1][3])
	
	# beliefs = [prior, belief1, belief2, belief3, belief4]
	# for b in beliefs:
	# 	map = concept_space.get_att_map_arr(concepts, b)
	# 	concept_space.arr2im(map).show()
	
	# print(concept_space.concepts_to_dscps(concepts))
