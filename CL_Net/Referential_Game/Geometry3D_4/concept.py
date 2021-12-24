import numpy as np
from collections import defaultdict
import random
import os
import pdb

'''
code book:
box: 0
sphere: 1
cylinder: 2
ellipsoid: 3
pyramid: 4
cone: 5
blue: 6
red: 7
yellow: 8
green: 9
cyan: 10
magenta: 11
small: 12
large: 13
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
	def __init__(self, game_config, is_train):
		self.attributes = game_config.attributes
		assert(len(self.attributes) == 18)
		self.attributes_np = np.array(self.attributes)
		self.shapes = list(range(6))
		self.colors = list(range(6, 12))
		self.sizes = list(range(12, 14))
		self.positions = list(range(14, 18))
		self.num_distractors = game_config.num_distractors
		self.num_distractors_range_np = np.array(range(self.num_distractors))
		self.concept_size = 4  # color, shape, size, position
		self.dataset_attributes_path = game_config.dataset_attributes_path
		self.img_h = game_config.img_h
		self.img_w = game_config.img_w
		self.save_dir = game_config.save_dir
		if not os.path.exists(self.dataset_attributes_path):
			print(self.dataset_attributes_path)
			raise Exception('dataset_attributes_path does not exist')
		self.dataset_attributes = np.load(self.dataset_attributes_path)
		assert(self.dataset_attributes.shape[1] == len(self.attributes))
		self.data_count = self.dataset_attributes.shape[0]
		if 'images_path' in game_config and game_config.images_path is not None:
			self.img_as_data = True
			self.images_path = game_config.images_path
			if not os.path.exists(self.images_path):
				print(self.images_path)
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
			self.teacher_feat_path = os.path.join(load_dir, 'Geometry3D_4_Teacher_features_%d_distractors.npy' % self.num_distractors)
			self.student_feat_path = os.path.join(load_dir, 'Geometry3D_4_Student_features_%d_distractors.npy' % self.num_distractors)
			if not os.path.exists(self.teacher_feat_path) or not os.path.exists(self.student_feat_path):
				print(self.teacher_feat_path, self.student_feat_path)
				raise Exception('teacher_feat_path or student_feat_path does not exist')
			self.teacher_features = np.load(self.teacher_feat_path)
			self.student_features = np.load(self.student_feat_path)
			assert(self.teacher_features.shape == self.student_features.shape)
			assert(self.teacher_features.shape[0] == self.data_count)
		
		self.permute = game_config.permutation_train if is_train else game_config.permutation_test
		if game_config.generated_dataset:
			self.generated_dataset = np.load(game_config.generated_dataset_path_train, allow_pickle = True).item() if is_train\
									 else np.load(game_config.generated_dataset_path_test, allow_pickle = True).item()
			self.dataset_size = len(self.generated_dataset['discrete_concepts'])
			self.fetch_idx = np.arange(self.dataset_size)
			self.num_used = np.random.randint(self.dataset_size)
		else:
			self.generated_dataset = None


	def store_features(self, teacher, student):
		def helper(agent):
			save_path = os.path.join(self.save_dir, 'Geometry3D_4_%s_features_%d_distractors.npy' % (agent.role_, self.num_distractors))
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
		if self.generated_dataset is not None:
			if self.num_used >= self.dataset_size:
				# np.random.shuffle(self.fetch_idx)
				self.num_used = 0
			chosen_idx = self.generated_dataset['discrete_concepts'][self.num_used]
		else:
			chosen_idx = np.random.choice(self.data_count, self.num_distractors, replace = False)
		concept_embed = self.dataset_attributes[chosen_idx]
		concepts = []
		included_attributes = set()
		for embed in concept_embed:
			concept = np.where(embed)[0].tolist()
			concepts.append(tuple(concept))
			included_attributes.update(concept)

		if self.generated_dataset is not None:
			stu_concept_idx = self.generated_dataset['stu_concept_idx'][self.num_used]
		else:
			stu_concept_idx = np.random.permutation(self.num_distractors)

		if self.permute:
			shape_permu = np.random.permutation(self.shapes)
			color_permu = np.random.permutation(self.colors)
			size_permu = np.random.permutation(self.sizes)
			position_permu = np.random.permutation(self.positions)
			if self.generated_dataset is not None:
				shuffle_mapping = self.generated_dataset['shuffle_mapping'][self.num_used]
			else:
				shuffle_mapping = np.concatenate([shape_permu, color_permu, size_permu, position_permu])
			reverse_shuffle_mapping = [p[0] for p in sorted(zip(self.attributes_np, shuffle_mapping), key = lambda x: x[1])]
			stu_concept_embed = concept_embed[stu_concept_idx, :][:, reverse_shuffle_mapping]
			stu_concepts = []
			stu_included_attributes = set()
			stu_chosen_idx = []
			for embed in stu_concept_embed:
				concept = np.where(embed)[0].tolist()
				stu_concepts.append(tuple(concept))
				concept_idx = concept[0] * 48 + (concept[1] - 6) * 8 + (concept[2] - 12) * 4 + (concept[3] - 14)
				stu_chosen_idx.append(concept_idx)
				stu_included_attributes.update(concept)
		else:
			stu_concepts = [concepts[i] for i in stu_concept_idx]
			stu_included_attributes = included_attributes
			stu_chosen_idx = chosen_idx[stu_concept_idx]
			shuffle_mapping = self.attributes_np
		
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

	def average_instance_share_check(self):
		from tqdm import tqdm
		self.num_distractors = 7
		counts = []
		training_set = np.load('/home/luyao/Datasets/Geometry3D_4/Geometry3D_4_%ddis_Train_1.npy' % self.num_distractors).item()
		test_set = np.load('/home/luyao/Datasets/Geometry3D_4/Geometry3D_4_%ddis_Test_1.npy' % self.num_distractors).item()
		training_concept_class = training_set['discrete_concepts']
		training_concept_class_set = []
		for cc in training_concept_class:
			training_concept_class_set.append(set(cc))
		
		test_concept_class = test_set['discrete_concepts'][:10000]
		for cc in tqdm(test_concept_class):
			cc_counts = []
			for cc_set_train in training_concept_class_set:
				count = 0
				for c in cc:
					if c in cc_set_train:
						count += 1
				cc_counts.append(count)
			counts.append(np.mean(cc_counts))
		
		print('mean count: ', np.mean(counts))

	
	def generate_training_dataset(self, size, novel_ratio = None, novel_concept = False, random_novel_concept = False):
		self.permute = True
		if not novel_concept:
			save_path = '/home/Datasets/Geometry3D_4/Geometry3D_4_%ddis_Train_2.npy' % self.num_distractors
			data = {'target_idx': [], 'shuffle_mapping': [], 'discrete_concepts': [], 'stu_concept_idx': []}
			for i in range(size):
				chosen_idx = np.random.choice(self.data_count, self.num_distractors, replace = False)
				shape_permu = np.random.permutation(self.shapes)
				color_permu = np.random.permutation(self.colors)
				size_permu = np.random.permutation(self.sizes)
				position_permu = np.random.permutation(self.positions)
				shuffle_mapping = np.concatenate([shape_permu, color_permu, size_permu, position_permu])
				stu_concept_idx = np.random.permutation(self.num_distractors)
				target_idx = np.random.randint(self.num_distractors)

				data['discrete_concepts'].append(chosen_idx)
				data['target_idx'].append(target_idx)
				data['shuffle_mapping'].append(shuffle_mapping)
				data['stu_concept_idx'].append(stu_concept_idx)
			for k in data:
				data[k] = np.array(data[k])
			np.save(save_path, data)
			return data
		else:
			if not random_novel_concept:
				separated_concepts = np.array([[0, 11, 13, 14], 
												[1, 10, 13, 14], 
												[2, 9, 13, 14], 
												[3, 8, 13, 14], 
												[4, 7, 13, 14], 
												[5, 6, 13, 14], 
												[0, 6, 12, 15], 
												[1, 7, 12, 15], 
												[2, 8, 12, 15], 
												[3, 9, 12, 15], 
												[4, 10, 12, 15], 
												[5, 11, 12, 15], 
												[0, 11, 12, 16], 
												[1, 10, 12, 16], 
												[2, 9, 12, 16], 
												[3, 8, 12, 16], 
												[4, 7, 12, 16], 
												[5, 6, 12, 16], 
												[0, 6, 13, 17], 
												[1, 7, 13, 17], 
												[2, 8, 13, 17], 
												[3, 9, 13, 17], 
												[4, 10, 13, 17], 
												[5, 11, 13, 17]])
				embeds = np.zeros((len(separated_concepts), len(self.attributes)))
				for row_i, col_j in enumerate(separated_concepts):
					embeds[row_i, col_j] = 1
				np.set_printoptions(threshold=np.inf, suppress=True, precision=3)
				separated_idx = []
				for embed in embeds:
					np_find = np.where(np.all(self.dataset_attributes == embed, axis = 1))
					assert(len(np_find) == 1 and len(np_find[0]) == 1)
					separated_idx.append(np_find[0][0])
				separated_idx_set = set(separated_idx)
				assert(len(separated_idx_set) == len(separated_idx))
				
			else:
				# continued = True
				# while continued:
				separated_idx = np.random.choice(self.data_count, round(novel_ratio * self.data_count), replace = False)
					# remain_idx = np.delete(np.arange(self.data_count), separated_idx)
					# remain_embed = self.dataset_attributes[remain_idx]
					# remain_att_count = np.sum(remain_embed, axis = 0)
					# if not np.all(remain_att_count >= 33):
					# 	continue
					# assert(len(separated_idx) == len(set(separated_idx)))
					# print(remain_att_count)
					# continued = False
			
			print(separated_idx)
			self.separated_idx = separated_idx
			self.separated_idx_set = set(separated_idx)
			train_sample_dist = np.ones(self.data_count)
			train_sample_dist[separated_idx] = 0
			train_sample_dist = train_sample_dist / np.sum(train_sample_dist)
			
			if random_novel_concept:
				save_path = '/home/Datasets/Geometry3D_4/Geometry3D_4_%ddis_Train_Novel_Concept_1.npy' % self.num_distractors
			else:
				save_path = '/home/Datasets/Geometry3D_4/Geometry3D_4_%ddis_Train_Fixed_Novel_Concept.npy' % self.num_distractors
			
			data = {'target_idx': [], 'shuffle_mapping': [], 'discrete_concepts': [], 'stu_concept_idx': []}
			for i in range(size):
				chosen_idx = np.random.choice(self.data_count, self.num_distractors, replace = False, p = train_sample_dist)
				# inner_continued = False
				# for idx in chosen_idx:
				# 	if idx in self.separated_idx_set:
				# 		raise Exception()
				# if inner_continued:
				# 	continue
				shape_permu = np.random.permutation(self.shapes)
				color_permu = np.random.permutation(self.colors)
				size_permu = np.random.permutation(self.sizes)
				position_permu = np.random.permutation(self.positions)
				shuffle_mapping = np.concatenate([shape_permu, color_permu, size_permu, position_permu])
				stu_concept_idx = np.random.permutation(self.num_distractors)
				target_idx = np.random.randint(self.num_distractors)

				data['discrete_concepts'].append(chosen_idx)
				data['target_idx'].append(target_idx)
				data['shuffle_mapping'].append(shuffle_mapping)
				data['stu_concept_idx'].append(stu_concept_idx)
				# i += 1
			for k in data:
				data[k] = np.array(data[k])
			np.save(save_path, data)
			return data
	

	def generate_novel_dataset(self):
		self.num_distractors = 7
		train_size = 600000
		test_size = 50000
		test_ratio = 0.2
		train_path = '/home/Datasets/Geometry3D_4/Geometry3D_4_%ddis_Train_Novel_Concept_3.npy' % self.num_distractors
		test_path = '/home/Datasets/Geometry3D_4/Geometry3D_4_%ddis_Test_Novel_Concept_3.npy' % self.num_distractors
		
		separated_idx = np.random.choice(self.data_count, round(test_ratio * self.data_count), replace = False)
		self.separated_idx_set = set(separated_idx)
		train_sample_dist = np.ones(self.data_count)
		train_sample_dist[separated_idx] = 0
		train_sample_dist = train_sample_dist / np.sum(train_sample_dist)
		
		train_data = {'target_idx': [], 'shuffle_mapping': [], 'discrete_concepts': [], 'stu_concept_idx': []}
		for i in range(train_size):
			chosen_idx = np.random.choice(self.data_count, self.num_distractors, replace = False, p = train_sample_dist)
			# inner_continued = False
			for idx in chosen_idx:
				if idx in self.separated_idx_set:
					raise Exception()
			# if inner_continued:
			# 	continue
			shape_permu = np.random.permutation(self.shapes)
			color_permu = np.random.permutation(self.colors)
			size_permu = np.random.permutation(self.sizes)
			position_permu = np.random.permutation(self.positions)
			shuffle_mapping = np.concatenate([shape_permu, color_permu, size_permu, position_permu])
			stu_concept_idx = np.random.permutation(self.num_distractors)
			target_idx = np.random.randint(self.num_distractors)

			train_data['discrete_concepts'].append(chosen_idx)
			train_data['target_idx'].append(target_idx)
			train_data['shuffle_mapping'].append(shuffle_mapping)
			train_data['stu_concept_idx'].append(stu_concept_idx)
			# i += 1
		for k in train_data:
			train_data[k] = np.array(train_data[k])
		np.save(train_path, train_data)

		test_data = {'target_idx': [], 'shuffle_mapping': [], 'discrete_concepts': [], 'stu_concept_idx': []}
		for i in range(test_size):
			chosen_idx = np.random.choice(separated_idx, self.num_distractors, replace = False)
			shape_permu = np.random.permutation(self.shapes)
			color_permu = np.random.permutation(self.colors)
			size_permu = np.random.permutation(self.sizes)
			position_permu = np.random.permutation(self.positions)
			shuffle_mapping = np.concatenate([shape_permu, color_permu, size_permu, position_permu])
			stu_concept_idx = np.random.permutation(self.num_distractors)
			target_idx = np.random.randint(self.num_distractors)
			test_data['discrete_concepts'].append(chosen_idx)
			test_data['target_idx'].append(target_idx)
			test_data['shuffle_mapping'].append(shuffle_mapping)
			test_data['stu_concept_idx'].append(stu_concept_idx)
		for k in test_data:
			test_data[k] = np.array(test_data[k])
		np.save(test_path, test_data)



	def generate_testing_dataset(self, size, novel_concept = False, random_novel_concept = False):
		self.permute = True
		if not novel_concept:
			training_set = np.load('../../../../Datasets/Geometry3D_4/Geometry3D_4_%ddis_Train_2.npy' % self.num_distractors).item()
			# if self.permute:
			training_teacher_concept_class = training_set['discrete_concepts']
			# training_shuffle_mapping = training_set['shuffle_mapping']
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
			
			data = {'target_idx': [], 'shuffle_mapping': [], 'discrete_concepts': [], 'stu_concept_idx': []}
			i = 0
			while i != size:
				chosen_idx = np.random.choice(self.data_count, self.num_distractors, replace = False)
				shape_permu = np.random.permutation(self.shapes)
				color_permu = np.random.permutation(self.colors)
				size_permu = np.random.permutation(self.sizes)
				position_permu = np.random.permutation(self.positions)
				shuffle_mapping = np.concatenate([shape_permu, color_permu, size_permu, position_permu])
				stu_concept_idx = np.random.permutation(self.num_distractors)
				target_idx = np.random.randint(self.num_distractors)

				sorted_p = []
				sorted_p.append(tuple(sorted(chosen_idx)))
				idx_mapped = sorted_p[0].index(chosen_idx[target_idx])
				sorted_p.append(idx_mapped)

				if tuple(sorted_p) in training_teacher_concept_class_set:
					continue

				data['discrete_concepts'].append(chosen_idx)
				data['target_idx'].append(target_idx)
				data['shuffle_mapping'].append(shuffle_mapping)
				data['stu_concept_idx'].append(stu_concept_idx)
				i += 1

			for k in data:
				data[k] = np.array(data[k])
			np.save('../../../../Datasets/Geometry3D_4/Geometry3D_4_%ddis_Test_2.npy' % self.num_distractors, data)
			return data
			# else:
			# 	training_teacher_concept_class = training_set['discrete_concepts']
			# 	training_target_idx = training_set['target_idx']
			# 	training_teacher_concept_class_new = []
			# 	for zip_pair in zip(training_teacher_concept_class, training_target_idx):
			# 		concept_class, idx = zip_pair
			# 		sorted_p = []
			# 		sorted_p.append(tuple(sorted(concept_class)))
			# 		idx_mapped = sorted_p[0].index(concept_class[idx])
			# 		sorted_p.append(idx_mapped)
			# 		training_teacher_concept_class_new.append(tuple(sorted_p))
			# 	training_teacher_concept_class_set = set(training_teacher_concept_class_new)
			# 	print('unique training set data: {}'.format(len(training_teacher_concept_class_set)))
				
			# 	data = {'target_idx': [], 'discrete_concepts': []}
			# 	i = 0
			# 	while i != size:
			# 		chosen_idx = np.random.choice(self.data_count, self.num_distractors, replace = False)
			# 		target_idx = np.random.randint(self.num_distractors)

			# 		sorted_p = []
			# 		sorted_p.append(tuple(sorted(chosen_idx)))
			# 		idx_mapped = sorted_p[0].index(chosen_idx[target_idx])
			# 		sorted_p.append(idx_mapped)

			# 		if tuple(sorted_p) in training_teacher_concept_class_set:
			# 			continue

			# 		data['discrete_concepts'].append(chosen_idx)
			# 		data['target_idx'].append(target_idx)
			# 		i += 1

			# 	for k in data:
			# 		data[k] = np.array(data[k])
			# 	np.save('/home/Datasets/Geometry3D_4/Geometry3D_4_%ddis_Test_No_Permutation.npy' % self.num_distractors, data)
			# 	return data
		else:
			data = {'target_idx': [], 'shuffle_mapping': [], 'discrete_concepts': [], 'stu_concept_idx': []}
			i = 0
			while i != size:
				target = np.random.choice(self.separated_idx)
				chosen_idx = np.random.choice(self.data_count, self.num_distractors - 1, replace = False)
				if target in chosen_idx:
					continue
				
				chosen_idx = np.random.permutation(np.append(chosen_idx, target))
				target_idx = chosen_idx.tolist().index(target)
				shape_permu = np.random.permutation(self.shapes)
				color_permu = np.random.permutation(self.colors)
				size_permu = np.random.permutation(self.sizes)
				position_permu = np.random.permutation(self.positions)
				shuffle_mapping = np.concatenate([shape_permu, color_permu, size_permu, position_permu])
				stu_concept_idx = np.random.permutation(self.num_distractors)

				data['discrete_concepts'].append(chosen_idx)
				data['target_idx'].append(target_idx)
				data['shuffle_mapping'].append(shuffle_mapping)
				data['stu_concept_idx'].append(stu_concept_idx)
				i += 1

			for k in data:
				data[k] = np.array(data[k])
			
			if random_novel_concept:
				np.save('/home/Datasets/Geometry3D_4/Geometry3D_4_%ddis_Test_Novel_Concept_1.npy' % self.num_distractors, data)
			else:
				np.save('/home/Datasets/Geometry3D_4/Geometry3D_4_%ddis_Test_Fixed_Novel_Concept.npy' % self.num_distractors, data)
			return data
			

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
				distractors = self.student_features
			else:
				raise Exception('Wrong role passed in generate_batch')
		for i in range(batch_size):
			chosen_idx = np.random.choice(self.data_count, self.num_distractors, replace = False)
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
	from easydict import EasyDict as edict
	from pprint import pprint as brint
	game_config = edict()
	game_config.num_distractors = 4
	game_config.message_space_size = 18
	game_config.attributes = range(18)
	game_config.n_grid_per_side = 2
	game_config.img_h = 128
	game_config.img_w = 128
	game_config.dataset_attributes_path = '../../../../Datasets/Geometry3D_4/Geometry3D_4_attributes.npy'
	game_config.generated_dataset = False
	game_config.generated_dataset_path_train = '../../../../Datasets/Geometry3D_4/Geometry3D_4_%ddis_Test_2.npy' % game_config.num_distractors
	game_config.feat_dir = '../../../../Datasets/Geometry3D_4/GeoFeat'
	game_config.save_dir = None
	game_config.permutation_train = True
	concept = Concept(game_config, True)
	concept.generate_training_dataset(600000, novel_ratio=0.2, novel_concept=True, random_novel_concept=True)
	concept.generate_testing_dataset(100000, novel_concept=True, random_novel_concept=True)

	# concept.average_instance_share_check()
	# concept.generate_training_dataset(600000)
	# concept.generate_testing_dataset(400000)
	# random_novel_concept = True
	# concept.generate_training_dataset(600000, novel_concept=True, random_novel_concept=random_novel_concept)
	# concept.generate_testing_dataset(400000, novel_concept=True, random_novel_concept=random_novel_concept)
	# for i in range(100000):
	# 	concept.rd_generate_concept()
		# input()
	# brint(np.array(concept.rd_generate_concept())[[0,1,4,5,6]])
	# np.set_printoptions(suppress=True, precision=5)
	
	
	# tea_cosine_maxs = []
	# stu_cosine_maxs = []
	# average = []
	# diffs = []
	# import time
	# time1 = time.time()
	# for i in range(60000):
	# 	_, _, tea_dis, stu_dis, _, stu_concept_idx, target_idx = concept.rd_generate_concept()
	# 	tea_target = tea_dis[target_idx]
	# 	tea_cosine = np.sum(tea_dis * tea_target, axis = 1) / (np.sqrt(np.sum(tea_dis ** 2, axis = 1)) * np.sqrt(np.sum(tea_target ** 2)))
	# 	stu_target = stu_dis[stu_concept_idx.tolist().index(target_idx)]
	# 	stu_cosine = np.sum(stu_dis * stu_target, axis = 1) / (np.sqrt(np.sum(stu_dis ** 2, axis = 1)) * np.sqrt(np.sum(stu_target ** 2)))

	# 	print(tea_cosine)
	# 	print(stu_cosine)
	# 	input()
	# 	tea_cosine_max = tea_cosine[np.argsort(tea_cosine)[2]]
	# 	stu_cosine_max = stu_cosine[np.argsort(stu_cosine)[2]]
	# 	diff = np.abs(tea_cosine_max - stu_cosine_max)

	# 	tea_cosine_maxs.append(tea_cosine_max)
	# 	stu_cosine_maxs.append(stu_cosine_max)
	# 	average.append((tea_cosine_max + stu_cosine_max) / 2)
	# 	diffs.append(diff)

	# 	# print(tea_cosine_max)
	# 	# print(stu_cosine_max)
	# 	# print(diff)

	# 	# input()
	# print('time: ', time.time() - time1)

	# import matplotlib.pyplot as plt
	# plt.hist(average, bins=50, normed=True)
	# plt.xlabel('average cosine')
	# plt.savefig('average_cosine.svg')
	# plt.show()
	# plt.hist(diffs, bins=50, normed=True)
	# plt.xlabel("diff between tea and stu's cosine")
	# plt.savefig('diff.svg')
	# plt.show()
	# plt.hist(tea_cosine_maxs, bins=50, normed=True)
	# plt.xlabel('teacher cosine')
	# plt.savefig('tea_cosine.svg')
	# plt.show()
	# plt.hist(stu_cosine_maxs, bins=50, normed=True)
	# plt.xlabel('student cosine')
	# plt.savefig('stu_cosine.svg')
	# plt.show()
