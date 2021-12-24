import numpy as np
from collections import defaultdict
import random
from tqdm import tqdm
import os

'''
code book:
black: 0
blue: 1
brown: 2
furry: 3
gray: 4
green: 5
long: 6
metallic: 7
orange: 8
pink: 9
rectangular: 10
red: 11
rough: 12
round: 13
shiny: 14
smooth: 15
spotted: 16
square: 17
striped: 18
vegetation: 19
violet: 20
wet: 21
white: 22
wooden: 23
yellow: 24
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
		assert(len(self.attributes) == 25)
		self.attributes_np = np.array(self.attributes)
		self.num_distractors = game_config.num_distractors
		self.num_distractors_range_np = np.array(range(self.num_distractors))
		self.concept_max_size = 25
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
			self.images = np.load(self.images_path, mmap_mode = 'r')
			assert(self.images.shape[1] == self.img_h and self.images.shape[2] == self.img_w)
			assert(self.images.shape[0] == self.data_count)
		else:
			if 'feat_dir' in game_config and game_config.feat_dir is not None:
				load_dir = game_config.feat_dir
			else:
				load_dir = self.save_dir
			self.img_as_data = False
			self.teacher_feat_path = os.path.join(load_dir, 'ImageNet_Teacher_features.npy')
			self.student_feat_path = os.path.join(load_dir, 'ImageNet_Student_features.npy')
			if not os.path.exists(self.teacher_feat_path) or not os.path.exists(self.student_feat_path):
				raise Exception('teacher_feat_path or student_feat_path does not exist')
			self.teacher_features = np.load(self.teacher_feat_path)
			self.student_features = np.load(self.student_feat_path)
			assert(self.teacher_features.shape == self.student_features.shape)
			assert(self.teacher_features.shape[0] == self.data_count)

		self.permute = game_config.permutation
		if game_config.generated_dataset:
			self.generated_dataset = np.load(game_config.generated_dataset_path).item()
			self.dataset_size = len(self.generated_dataset['discrete_concepts'])
			self.fetch_idx = np.arange(self.dataset_size)
			self.num_used = np.random.randint(self.dataset_size)
		else:
			self.generated_dataset = None



	def store_features(self, teacher, student):
		def helper(agent):
			save_path = os.path.join(self.save_dir, 'ImageNet_%s_features.npy' % agent.role_)
			batch_size = self.num_distractors * 2
			feature_batches = []
			for i in tqdm(range(0, self.images.shape[0], batch_size)):
				curr_batch = self.images[i: i + batch_size]
				mod = curr_batch.shape[0] % self.num_distractors
				if mod == 0:
					cnn_input = curr_batch.reshape([-1, self.num_distractors, *self.images.shape[1:]])
				else:
					dim_to_append = self.num_distractors - mod
					padding = np.zeros([dim_to_append, *self.images.shape[1:]])
					cnn_input = np.concatenate([curr_batch, padding], axis = 0).reshape([-1, self.num_distractors, *self.images.shape[1:]])
				cnn_output = agent.sess_.run(agent.perception_core_.visual_features_, feed_dict = {agent.distractors_: cnn_input})
				if mod == 0:
					features = cnn_output.reshape([curr_batch.shape[0], cnn_output.shape[-1]])
				else:
					features = cnn_output.reshape([curr_batch.shape[0] + dim_to_append, cnn_output.shape[-1]])[:curr_batch.shape[0]]
				feature_batches.append(features)
			np.save(save_path, np.concatenate(feature_batches, axis = 0))
		helper(teacher)
		helper(student)


	def rd_generate_concept(self):
		if self.generated_dataset is not None:
			if self.num_used >= self.dataset_size:
				self.num_used = 0
			chosen_idx = self.generated_dataset['discrete_concepts'][self.num_used]
			concept_embed = self.dataset_attributes[chosen_idx]
			concepts = []
			included_attributes = set()
			for embed in concept_embed:
				sparse = np.where(embed)[0].tolist()
				concepts.append(tuple(sparse))
				included_attributes.update(sparse)
			assert(len(set(concepts)) == len(concepts))
		else:
			continued = True
			while continued:
				chosen_idx = np.random.choice(self.data_count, self.num_distractors, replace = False)
				concept_embed = self.dataset_attributes[chosen_idx]
				concepts = []
				included_attributes = set()
				for embed in concept_embed:
					sparse = np.where(embed)[0].tolist()
					concepts.append(tuple(sparse))
					included_attributes.update(sparse)
				if len(set(concepts)) == len(concepts):
					continued = False
		
		shuffle_mapping = self.attributes_np
		if self.generated_dataset is not None:
			stu_concept_idx = self.generated_dataset['stu_concept_idx'][self.num_used]
		else:
			stu_concept_idx = np.random.permutation(self.num_distractors)
		
		stu_concept_embed = concept_embed[stu_concept_idx, :]
		stu_chosen_idx = chosen_idx[stu_concept_idx]
		stu_concepts = []
		for embed in stu_concept_embed:
			concept = np.where(embed)[0].tolist()
			stu_concepts.append(tuple(concept))
		
		stu_included_attributes = included_attributes

		if self.img_as_data:
			distractors = self.images[chosen_idx]
			stu_distractors = self.images[stu_chosen_idx]
		else:
			distractors = self.teacher_features[chosen_idx]
			stu_distractors = self.student_features[stu_chosen_idx]
		
		if self.generated_dataset is not None:
			target_idx = self.generated_dataset['target_idx'][self.num_used]
			self.num_used += 1
		else:
			target_idx = np.random.randint(self.num_distractors)
		
		return (concepts, stu_concepts), (list(included_attributes), list(stu_included_attributes)), \
			distractors, stu_distractors, shuffle_mapping, stu_concept_idx, target_idx


	def teaching_dim(self, concepts, included_attrs):
		td_dict = {}
		teaching_sample = defaultdict(list)
		sample_size = 1
		smallest_sample_size = self.concept_max_size
		for i in range(len(concepts)):
			for j in range(len(concepts)):
				if set(concepts[i]).issubset(set(concepts[j])) and i != j:
					td_dict[tuple(concepts[i])] = (self.concept_max_size, tuple(concepts[i]))
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
	
	def teaching_level_for_each_concept(self, concepts):
		levels = np.ones(self.num_distractors).astype(int)
		non_ui_indicator = np.ones(self.num_distractors).astype(int)
		check_ui = np.zeros(self.num_distractors)
		if self.recursive_teaching_dim(concepts) > 1:
			levels = np.zeros(self.num_distractors).astype(int)
		else:
			while len([x for y in concepts for x in y]) > 0:
				for idx in range(self.num_distractors):
					included_attributes = []
					for i in range(len(concepts)):
						if i != idx:
							for e in concepts[i]:
								included_attributes.append(e)
					set_other = (set(included_attributes))
					set_all = (set([x for y in concepts for x in y ]))	
					check_ui[idx] =  (set_other == set_all)		 
				non_ui_indicator = np.multiply(non_ui_indicator, check_ui).astype(int)
				levels += non_ui_indicator
				new_concepts = []
				for i in range(4):
					if non_ui_indicator[i] == 0:
						new_concepts.append(())
					else:
						new_concepts.append(concepts[i])
				concepts = new_concepts
				#concepts = [c for c in (concepts * non_ui_indicator)]
				#pdb.set_trace()
		return levels


	def bayesian_update(self, old_belief, concepts, info):
		likelihood = []
		for concept in concepts:
			prob = 1.0 * (info in concept) / len(concept)
			likelihood.append(prob)
		new_belief = old_belief * np.array(likelihood)
		new_belief /= np.sum(new_belief) + 1e-9
		return new_belief

	
	def generate_training_dataset(self, size):
		save_path = '/home/Datasets/ImageNet/ImageNet_%ddis_Train.npy' % self.num_distractors
		data = {'target_idx': [], 'discrete_concepts': [], 'stu_concept_idx': []}
		for i in range(size):
			continued = True
			while continued:
				chosen_idx = np.random.choice(self.data_count, self.num_distractors, replace = False)
				concept_embed = self.dataset_attributes[chosen_idx]
				concepts = []
				included_attributes = set()
				for embed in concept_embed:
					sparse = np.where(embed)[0].tolist()
					concepts.append(tuple(sparse))
					included_attributes.update(sparse)
				if len(set(concepts)) == len(concepts):
					continued = False
			stu_concept_idx = np.random.permutation(self.num_distractors)
			target_idx = np.random.randint(self.num_distractors)
			data['discrete_concepts'].append(chosen_idx)
			data['stu_concept_idx'].append(stu_concept_idx)
			data['target_idx'].append(target_idx)
		for k in data:
			data[k] = np.array(data[k])
		np.save(save_path, data)
		return data


	def generate_testing_dataset(self, size):
		training_set = np.load('/home/Datasets/ImageNet/ImageNet_%ddis_Train.npy' % self.num_distractors).item()
		training_teacher_concept_class = training_set['discrete_concepts']
		training_target_idx = training_set['target_idx']
		training_teacher_concept_class_new = []
		for zip_pair in zip(training_teacher_concept_class, training_target_idx):
			concept_class, idx = zip_pair
			sorted_p = []
			sorted_p.append(tuple(sorted(concept_class)))
			idx_mapped = sorted_p[0].index(concept_class[idx])
			sorted_p.append(idx_mapped)
			training_teacher_concept_class_new.append(tuple(sorted_p))
		training_teacher_concept_class_set = set(training_teacher_concept_class_new)
		print('unique training set data: {}'.format(len(training_teacher_concept_class_set)))
		
		data = {'target_idx': [], 'stu_concept_idx':[], 'discrete_concepts': []}
		i = 0
		while i != size:
			continued = True
			while continued:
				chosen_idx = np.random.choice(self.data_count, self.num_distractors, replace = False)
				concept_embed = self.dataset_attributes[chosen_idx]
				concepts = []
				included_attributes = set()
				for embed in concept_embed:
					sparse = np.where(embed)[0].tolist()
					concepts.append(tuple(sparse))
					included_attributes.update(sparse)
				if len(set(concepts)) == len(concepts):
					continued = False
			stu_concept_idx = np.random.permutation(self.num_distractors)
			target_idx = np.random.randint(self.num_distractors)

			sorted_p = []
			sorted_p.append(tuple(sorted(chosen_idx)))
			idx_mapped = sorted_p[0].index(chosen_idx[target_idx])
			sorted_p.append(idx_mapped)

			if tuple(sorted_p) in training_teacher_concept_class_set:
				continue

			data['discrete_concepts'].append(chosen_idx)
			data['stu_concept_idx'].append(stu_concept_idx)
			data['target_idx'].append(target_idx)
			i += 1

		for k in data:
			data[k] = np.array(data[k])
		np.save('/home/Datasets/ImageNet/ImageNet_%ddis_Test.npy' % self.num_distractors, data)
		return data
		

	def generate_batch(self, batch_size, role, epsilon = 0.5):
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
				distractors = self.student_features
			else:
				raise Exception('Wrong role passed in generate_batch')
		for i in range(batch_size):
			continued = True
			while continued:
				chosen_idx = np.random.choice(self.data_count, self.num_distractors, replace = False)
				concept_embed = self.dataset_attributes[chosen_idx]
				concepts = []
				included_attributes = set()
				for embed in concept_embed:
					sparse = np.where(embed)[0].tolist()
					included_attributes.update(sparse)
					concepts.append(tuple(sparse))
				if len(set(concepts)) == len(concepts):
					continued = False
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
	def visualize(chosen_idx):
		import cv2
		codebook = {0:'black', 
		1:'blue', 
		2:'brown', 
		3:'furry', 
		4:'gray', 
		5:'green', 
		6:'long', 
		7:'metallic', 
		8:'orange', 
		9:'pink', 
		10:'rectangular', 
		11:'red', 
		12:'rough', 
		13:'round', 
		14:'shiny', 
		15:'smooth', 
		16:'spotted', 
		17:'square', 
		18:'striped', 
		19:'vegetation', 
		20:'violet', 
		21:'wet', 
		22:'white', 
		23:'wooden', 
		24:'yellow'}
		codemap = lambda x: codebook[x]
		images = np.load('ImageNet_images.npy', mmap_mode = 'r')
		# print(images.shape)
		attr = np.load('ImageNet_attributes.npy')
		# print(attr.shape)
		# for i in np.random.randint(0, 6762, size=20):
		# for i in range(0, 6762):
		for i in chosen_idx:
			concept = np.where(attr[i])[0]
			print(concept)
			print(list(map(codemap, concept)))
			norm_image = cv2.normalize(images[i][:,:,::-1], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) 
			cv2.imshow('img', norm_image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
	
	visualize([5214, 5748, 2279, 2510, 5885, 2623,  775, 2279])
	exit()


	def td(num_dis):
		import pdb
		def teaching_dim(concepts, included_attrs):
			td_dict = {}
			teaching_sample = defaultdict(list)
			sample_size = 1
			smallest_sample_size = 25
			for i in range(len(concepts)):
				for j in range(len(concepts)):
					if set(concepts[i]).issubset(set(concepts[j])) and i != j:
						td_dict[tuple(concepts[i])] = (25, tuple(concepts[i]))
			while len(td_dict) < len(concepts):
				all_teaching_samples = get_combination(included_attrs, sample_size)
				if sample_size > 25:
					pdb.set_trace()
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
		attr = np.load('ImageNet_attributes.npy')
		size = attr.shape[0]


		for i in range(100000):
			continued = True
			while continued:
				chosen_idx = np.random.randint(0, size, size = num_dis)
				concept_embed = attr[chosen_idx]
				concepts = []
				included_attributes = set()
				for embed in concept_embed:
					sparse = np.where(embed)[0].tolist()
					included_attributes.update(sparse)
					concepts.append(tuple(sparse))
				if len(set(concepts)) == len(concepts):
					continued = False
			td_dict = teaching_dim(concepts, list(included_attributes))
			# print(td_dict)
	
	# td(4)

	from easydict import EasyDict as edict
	from pprint import pprint as brint
	game_config = edict()
	game_config.num_distractors = 4
	game_config.message_space_size = 25
	game_config.attributes = range(25)
	game_config.img_h = 224
	game_config.img_w = 224
	game_config.bbox_ratio_thred = 0.5
	game_config.dataset_attributes_path = 'ImageNet_attributes.npy'
	game_config.permutation = False
	game_config.images_path = 'ImageNet_images.npy'
	game_config.save_dir = None
	game_config.generated_dataset = False
	concept_space = Concept(game_config)
	# # concept.generate_training_dataset(600000)
	# # concept.generate_testing_dataset(400000)
	# for _ in range(10):
	# 	brint(np.array(concept.rd_generate_concept())[[0,1,4,5,6]].tolist())
	# 	input()


	level_map = dict()
	for i in range(100000):
		continued = True
		while continued:
			chosen_idx = np.random.randint(0, concept_space.data_count, size = game_config.num_distractors)
			concept_embed = concept_space.dataset_attributes[chosen_idx]
			concepts = []
			included_attributes = set()
			for embed in concept_embed:
				sparse = np.where(embed)[0].tolist()
				included_attributes.update(sparse)
				concepts.append(tuple(sparse))
			if len(set(concepts)) == len(concepts):
				continued = False
		# print(concepts)
		# print(concept_space.teaching_level_for_each_concept(concepts))
		# input()
		find_level2 = np.where(concept_space.teaching_level_for_each_concept(concepts) >= 2)[0]
		level2_indices = chosen_idx[find_level2]
		for level2_idx in level2_indices:
			if level2_idx in level_map:
				print(level2_idx)
				stored_chosen_idx = level_map[level2_idx][0]
				print(level_map[level2_idx])
				print((chosen_idx, concepts))
				visualize(np.concatenate([stored_chosen_idx, chosen_idx]))
				del level_map[level2_idx]
			else:
				level_map[level2_idx] = (chosen_idx, concepts)
