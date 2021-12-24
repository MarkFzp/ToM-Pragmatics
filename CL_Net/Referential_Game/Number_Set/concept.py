import time
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from easydict import EasyDict as edict
# import matplotlib.pyplot as plt
import pickle
import pdb

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
		self.attributes_np = np.array(self.attributes)
		self.num_distractors = game_config.num_distractors
		self.num_distractors_range_np = np.array(range(self.num_distractors))
		self.concept_max_size = game_config.concept_max_size
		self.permute = game_config.permutation_train if is_train else game_config.permutation_test
		if game_config.generated_dataset:
			self.generated_dataset = np.load(game_config.generated_dataset_path_train, allow_pickle = True).item() if is_train\
									 else np.load(game_config.generated_dataset_path_test, allow_pickle = True).item()
			self.dataset_size = len(self.generated_dataset['discrete_concepts'])
			self.num_used = np.random.randint(self.dataset_size)
			self.fetch_idx = np.arange(self.dataset_size)
		else:
			self.generated_dataset = None

	def rd_generate_concept(self):
		if self.generated_dataset is not None:
			if self.num_used >= self.dataset_size:
				# np.random.shuffle(self.fetch_idx)
				self.num_used = 0
			concepts = self.generated_dataset['discrete_concepts'][self.num_used].tolist()
			included_attributes = set()
			for concept in concepts:
				included_attributes.update(concept)
			included_attributes = list(included_attributes)
		else:
			concepts = set()
			included_attributes = set()
			while len(concepts) < self.num_distractors:
				concept_idx = np.sort(np.random.choice(len(self.attributes), self.concept_max_size))
				concept = tuple(set([self.attributes[ci] for ci in concept_idx]))
				concepts.add(concept)
				for attr in concept:
					included_attributes.add(attr)
			concepts = list(concepts)
			included_attributes = list(included_attributes)
		
		concept_embed = np.zeros((self.num_distractors, len(self.attributes)))
		for idx, concept in enumerate(concepts):
			concept_embed[idx, concept] = 1
	
		if self.generated_dataset is not None:
			stu_concept_idx = self.generated_dataset['stu_concept_idx'][self.num_used]
		else:
			stu_concept_idx = np.random.permutation(self.num_distractors)

		if self.permute:
			if self.generated_dataset is not None:
				shuffle_mapping = self.generated_dataset['shuffle_mapping'][self.num_used]
			else:
				shuffle_mapping = np.random.permutation(self.attributes)
			
			stu_concepts_col = []
			for concept in concepts:
				stu_concepts_col.append(tuple([shuffle_mapping[i] for i in concept]))
			
			stu_included_attributes = [shuffle_mapping[i] for i in included_attributes]
			
		else:
			stu_concepts_col = concepts
			shuffle_mapping = self.attributes_np
			stu_included_attributes = included_attributes

		stu_concepts = []
		for idx, _ in enumerate(stu_concepts_col):
			stu_concepts.append(stu_concepts_col[stu_concept_idx[idx]])
		
		stu_concept_embed = np.zeros((self.num_distractors, len(self.attributes)))
		for idx, concept in enumerate(stu_concepts):
			stu_concept_embed[idx, stu_concepts[idx]] = 1

		if self.generated_dataset is not None:
			target_idx = self.generated_dataset['target_idx'][self.num_used]
			self.num_used += 1
		else:
			target_idx = np.random.randint(self.num_distractors)

		return (concepts, stu_concepts), (included_attributes, stu_included_attributes),\
			concept_embed, stu_concept_embed,\
			shuffle_mapping, stu_concept_idx, target_idx

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
			sample_size += 1
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
	
	def instance_share_check(self):
		from tqdm import tqdm
		self.num_distractors = 7
		shared_count = [0 for _ in range(self.num_distractors + 1)]
		training_set = np.load('/home/luyao/Datasets/Number_Set/Number_Set_%ddis_Train_2.npy' % self.num_distractors).item()
		test_set = np.load('/home/luyao/Datasets/Number_Set/Number_Set_%ddis_Test_2.npy' % self.num_distractors).item()
		training_concept_class = training_set['discrete_concepts']
		training_concept_class_set = []
		for cc in training_concept_class:
			training_concept_class_set.append(set([tuple(sorted(c)) for c in cc]))
		
		test_concept_class = test_set['discrete_concepts'][:10000]
		for cc in tqdm(test_concept_class):
			max_count = 0
			cc_tuple = [tuple(sorted(c)) for c in cc]
			for cc_set_train in training_concept_class_set:
				count = 0
				for c_tuple in cc_tuple:
					if c_tuple in cc_set_train:
						count += 1
				if count > max_count:
					max_count = count
			shared_count[max_count] += 1
		
		shared_count_percent = np.array(shared_count) / len(test_concept_class)
		print(shared_count_percent)
		return shared_count_percent
	

	def average_instance_share_check(self):
		from tqdm import tqdm
		self.num_distractors = 7
		counts = []
		training_set = np.load('/home/luyao/Datasets/Number_Set/Number_Set_%ddis_Train_2.npy' % self.num_distractors).item()
		test_set = np.load('/home/luyao/Datasets/Number_Set/Number_Set_%ddis_Test_2.npy' % self.num_distractors).item()
		training_concept_class = training_set['discrete_concepts']
		training_concept_class_set = []
		for cc in training_concept_class:
			training_concept_class_set.append(set([tuple(sorted(c)) for c in cc]))
		
		test_concept_class = test_set['discrete_concepts'][:10000]
		for cc in tqdm(test_concept_class):
			cc_counts =[]
			cc_tuple = [tuple(sorted(c)) for c in cc]
			for cc_set_train in training_concept_class_set:
				count = 0
				for c_tuple in cc_tuple:
					if c_tuple in cc_set_train:
						count += 1
				cc_counts.append(count)
			counts.append(np.mean(cc_counts))
		
		print('mean count: ', np.mean(counts))


	def generate_novel_dataset(self, test_concept_portion, train_set_size, test_set_size):
		from scipy.special import comb
		import itertools
	
		self.permute = True
		train_path = 'Number_Set_%ddis_Train_3.npy' % self.num_distractors
		test_path = 'Number_Set_%ddis_Test_3.npy' % self.num_distractors

		total_concept_sizes = [comb(len(self.attributes), i, exact = True) for i in range(1, self.concept_max_size + 1)]
		total_concept_idxs = []
		for i, size in enumerate(total_concept_sizes):
			if i == 0:
				total_concept_idxs.append(size)
			else:
				total_concept_idxs.append(total_concept_idxs[-1] + size)

		total_concept_size = np.sum(total_concept_sizes)
		assert(total_concept_idxs[-1] == total_concept_size)
		# test_concept_size = int(round(total_concept_size * test_concept_portion))
		all_concept = np.array(list(itertools.chain.from_iterable([list(itertools.combinations(range(len(self.attributes)), i)) for i in range(1, self.concept_max_size + 1)])))

		assert(len(set(all_concept)) == len(all_concept) == total_concept_size)
		print(all_concept)

		test_set_idx = []
		for i, idx in enumerate(total_concept_idxs):
			if i == 0:
				test_set_idx += np.random.choice(idx, size = int(round(test_concept_portion * idx)), replace = False).tolist()
			else:
				test_set_idx += np.random.choice(range(total_concept_idxs[i - 1], idx), size = int(round(test_concept_portion * (idx - total_concept_idxs[i - 1]))), replace = False).tolist()
		
		print(test_set_idx)
		# test_set_idx = np.random.choice(total_concept_size, size = test_concept_size, replace = False)
		train_set_idx = list(set(range(total_concept_size)) - set(test_set_idx))

		if self.concept_max_size == 4:
			weight_map = [0.04153, 0.583, 2.0469, 1]
		elif self.concept_max_size == 5:
			weight_map = [0.008918, 0.247, 1.249, 1.996, 1]
		else:
			raise Exception()
		
		train_set_idx_prob = np.array([weight_map[len(all_concept[idx]) - 1] for idx in train_set_idx])
		train_set_idx_prob = train_set_idx_prob / np.sum(train_set_idx_prob)
		test_set_idx_prob = np.array([weight_map[len(all_concept[idx]) - 1] for idx in test_set_idx])
		test_set_idx_prob = test_set_idx_prob / np.sum(test_set_idx_prob)
		print(train_set_idx_prob)
		print(test_set_idx_prob)

		train_data = {'target_idx': [], 'shuffle_mapping': [], 'discrete_concepts': [], 'stu_concept_idx': []}
		for _ in range(train_set_size):
			discrete_concept_idx = np.random.choice(train_set_idx, size = self.num_distractors, replace = False, p = train_set_idx_prob)
			discrete_concept = all_concept[discrete_concept_idx].tolist()
			shuffle_mapping = np.random.permutation(self.attributes)
			stu_concept_idx = np.random.permutation(self.num_distractors)
			target_idx = np.random.choice(self.num_distractors)
			train_data['target_idx'].append(target_idx)
			train_data['shuffle_mapping'].append(shuffle_mapping)
			train_data['stu_concept_idx'].append(stu_concept_idx)
			train_data['discrete_concepts'].append(discrete_concept)
		for k in train_data:
			train_data[k] = np.array(train_data[k])
		
		test_data = {'target_idx': [], 'shuffle_mapping': [], 'discrete_concepts': [], 'stu_concept_idx': []}
		for _ in range(test_set_size):
			discrete_concept_idx = np.random.choice(test_set_idx, size = self.num_distractors, replace = False, p = test_set_idx_prob)
			discrete_concept = all_concept[discrete_concept_idx].tolist()
			shuffle_mapping = np.random.permutation(self.attributes)
			stu_concept_idx = np.random.permutation(self.num_distractors)
			target_idx = np.random.choice(self.num_distractors)
			test_data['target_idx'].append(target_idx)
			test_data['shuffle_mapping'].append(shuffle_mapping)
			test_data['stu_concept_idx'].append(stu_concept_idx)
			test_data['discrete_concepts'].append(discrete_concept)
		for k in test_data:
			test_data[k] = np.array(test_data[k])

		np.save(train_path, train_data)
		np.save(test_path, test_data)
		return train_data, test_data







	def generate_training_dataset(self, size):
		self.permute = True
		save_path = '/home/Datasets/Number_Set/Number_Set_%ddis_12attr_5max_Train_3.npy' % self.num_distractors
		data = {'target_idx': [], 'shuffle_mapping': [], 'discrete_concepts': [], 'stu_concept_idx': []}
		for i in range(size):
			discrete_concepts, included_attributes, concept_embed, stu_concept_embed, \
				shuffle_mapping, stu_concept_idx, target_idx = self.rd_generate_concept()
			data['target_idx'].append(target_idx)
			data['shuffle_mapping'].append(shuffle_mapping)
			data['stu_concept_idx'].append(stu_concept_idx)
			data['discrete_concepts'].append(discrete_concepts[0])
		for k in data:
			data[k] = np.array(data[k])
		np.save(save_path, data)
		return data

	def generate_testing_dataset(self, size):
		self.permute = True
		training_set = np.load('../../../../Datasets/Number_Set/Number_Set_%ddis_Train_3.npy' % self.num_distractors).item()
		training_teacher_concept_class = training_set['discrete_concepts']
		# training_shuffle_mapping = training_set['shuffle_mapping']
		training_target_idx = training_set['target_idx']
		training_teacher_concept_class_new = []
		for concept_class, idx in zip(training_teacher_concept_class, training_target_idx):
			sorted_p = []
			sorted_p.append(tuple(sorted([tuple(sorted(c)) for c in concept_class])))
			idx_mapped = sorted_p[0].index(tuple(sorted(concept_class[idx])))
			sorted_p.append(idx_mapped)
			training_teacher_concept_class_new.append(tuple(sorted_p))
		training_teacher_concept_class_set = set(training_teacher_concept_class_new)
		print('unique training set data: {}'.format(len(training_teacher_concept_class_set)))
		

		data = {'target_idx': [], 'shuffle_mapping': [], 'discrete_concepts': [], 'stu_concept_idx': []}
		i = 0
		while i != size:
			discrete_concepts, included_attributes, concept_embed, stu_concept_embed, \
				shuffle_mapping, stu_concept_idx, target_idx = self.rd_generate_concept()
			
			teacher_concept_class = discrete_concepts[0]
			discrete_concepts_sorted = []
			discrete_concepts_sorted.append(tuple(sorted([tuple(sorted(c)) for c in teacher_concept_class])))
			target_idx_mapped = discrete_concepts_sorted[0].index(tuple(sorted(teacher_concept_class[target_idx])))
			discrete_concepts_sorted.append(target_idx_mapped)
			if tuple(discrete_concepts_sorted) in training_teacher_concept_class_set:
				print('big new: collide with training set!!!!!!')
				continue
			data['target_idx'].append(target_idx)
			data['shuffle_mapping'].append(shuffle_mapping)
			data['stu_concept_idx'].append(stu_concept_idx)
			data['discrete_concepts'].append(teacher_concept_class)
			i += 1
		for k in data:
			data[k] = np.array(data[k])
		np.save('../../../../Datasets/Number_Set/Number_Set_%ddis_Test_2.npy' % self.num_distractors, data)
		return data
	
	#generate a datasets of 4 types: whole dataset, td > 1, rtd = 1, td > 1 && rtd = 1
	# def generate_fixed_datasets(self, num_of_games, equiv = True):
		# t_data = []
		# t_data_td = []
		# s_data = []
		# s_data_td = []
		# attr_equiv_class_mappings = []
		# t_s_concept_idx_mappings = []

		# t_hard_data = []
		# t_hard_target_indices = []
		# s_hard_data = []
		# s_hard_target_indices = []
		# # hard_attr_equiv_class_mappings = []
		# # hard_t_s_concept_idx_mappings = []

		# t_rtd_data = []
		# t_rtd_data_td = []
		# s_rtd_data = []
		# s_rtd_data_td = []
		# rtd_attr_equiv_class_mappings = []
		# rtd_t_s_concept_idx_mappings = []

		# t_rtd_hard_data = []
		# t_rtd_hard_target_indices = []
		# s_rtd_hard_data = []
		# s_rtd_hard_target_indices = []
		# # rtd_hard_attr_equiv_class_mappings = []
		# # rtd_hard_t_s_concept_idx_mappings = []


		# for _ in range(num_of_games):
		# 	t_concept_to_embed_dict = {}
		# 	s_concept_to_embed_dict = {}
		# 	(t_concepts,s_concepts), (t_included_attr, s_included_attr),\
		# 	t_concept_embed, s_concept_embed, \
		# 	attr_equiv_class_mapping, t_s_concept_idx_mapping  = self.rd_generate_concept()
			
		# 	attr_equiv_class_mappings.append(attr_equiv_class_mapping)
		# 	t_s_concept_idx_mappings.append(t_s_concept_idx_mapping)
			
		# 	#generate mapping from concept to concept embedding
		# 	for i in range(self.num_distractors):
		# 		t_concept_to_embed_dict[t_concepts[i]] = t_concept_embed[i]
		# 		s_concept_to_embed_dict[s_concepts[i]] = s_concept_embed[i]
			
		
		# 	#find teaching dimension of each concept in this game
		# 	# t_td_dict, _ = self.teaching_dim(t_concepts, t_included_attr)
		# 	# s_td_dict, _ = self.teaching_dim(s_concepts, s_included_attr)

		# 	#generate data and their teaching dimension for teacher and students
		# 	for i in range(self.num_distractors):
		# 		t_data.append(t_concept_to_embed_dict[t_concepts[i]]) 
		# 		s_data.append(s_concept_to_embed_dict[s_concepts[i]])
		# 		# t_data_td.append(t_td_dict[t_concepts[i]][0])
		# 		# s_data_td.append(s_td_dict[s_concepts[i]][0])
			
					
		# 	#generate data and their teaching dimension for rtd=1 problem
		# 	# if self.recursive_teaching_dim(t_concepts) == 1:
		# 	# 	if self.recursive_teaching_dim(s_concepts) != 1:
		# 	# 		print("teacher and student's rtd not same!")
		# 	# 		raise Exception()
				
		# 	# 	for i in range(self.num_concepts):
		# 	# 		t_rtd_data.append(t_concept_to_embed_dict[t_concepts[i]])
		# 	# 		t_rtd_data_td.append(t_td_dict[t_concepts[i]][0])        
		# 	# 		s_rtd_data.append(s_concept_to_embed_dict[s_concepts[i]])
		# 	# 		s_rtd_data_td.append(s_td_dict[s_concepts[i]][0])
				
		# 	# 	rtd_attr_equiv_class_mappings.append(attr_equiv_class_mapping)
		# 	# 	rtd_t_s_concept_idx_mappings.append(t_s_concept_idx_mapping)

		

		# # num_rtd_data = int(len(t_rtd_data) / self.num_concepts)
		# # num_rtd_data_temp = int(len(s_rtd_data) / self.num_concepts)
		# # assert(num_rtd_data == num_rtd_data_temp)

		# #generate student & teacher target indices for whole dataset and rtd = 1 problem 
		# t_s_concept_idx_mappings = np.array(t_s_concept_idx_mappings)
		# s_target_indices = np.random.choice(range(self.num_distractors), num_of_games)
		# t_target_indices = t_s_concept_idx_mappings[np.arange(len(s_target_indices)), s_target_indices]
	
		# # rtd_t_s_concept_idx_mappings = np.array(rtd_t_s_concept_idx_mappings)
		# # s_rtd_target_indices = np.random.choice(range(self.num_concepts), num_rtd_data)
		# # t_rtd_target_indices = rtd_t_s_concept_idx_mappings[np.arange(len(s_rtd_target_indices)),s_rtd_target_indices]
	

		# #generate hard problems for whole test set(teaching dimension > 1)
		# # for i in range(num_of_games):
		# # 	if(t_data_td[i * self.num_concepts + t_target_indices[i]] > 1):
		# # 		if(s_data_td[i * self.num_concepts + s_target_indices[i]] <= 1):
		# # 			print("teacher and student's td > 1 problem not same!")
		# # 			raise Exception()
		# # 		for j in range(self.num_concepts):
		# # 			t_hard_data.append(t_data[i * self.num_concepts + j])
		# # 			s_hard_data.append(s_data[i * self.num_concepts + j])

		# # 		t_hard_target_indices.append(t_target_indices[i])
		# # 		s_hard_target_indices.append(s_target_indices[i])


		# #generate hard problems for rtd = 1 test set
		# # for i in range(num_rtd_data):
		# # 	if(t_rtd_data_td[i * self.num_concepts + t_rtd_target_indices[i]] > 1):
		# # 		if(s_rtd_data_td[i * self.num_concepts + s_rtd_target_indices[i]] <= 1):
		# # 			print("teacher and student's td > 1 & rtd = 1problem not same!")
		# # 			raise Exception()
		# # 		for j in range(self.num_concepts):
		# # 			t_rtd_hard_data.append(t_rtd_data[i * self.num_concepts + j])
		# # 			s_rtd_hard_data.append(s_rtd_data[i * self.num_concepts + j])

		# # 		t_rtd_hard_target_indices.append(t_rtd_target_indices[i])
		# # 		s_rtd_hard_target_indices.append(s_rtd_target_indices[i])


		# t_data = np.array(t_data)
		# # t_hard_data = np.array(t_hard_data)
		# # t_hard_target_indices = np.array(t_hard_target_indices)
		# # t_num_hard_data = len(t_hard_target_indices)

		# # t_rtd_data = np.array(t_rtd_data)
		# # t_rtd_hard_data = np.array(t_rtd_hard_data)
		# # t_rtd_hard_target_indices = np.array(t_rtd_hard_target_indices)
		# # t_num_rtd_hard_data = len(t_rtd_hard_target_indices)

		# s_data = np.array(s_data)
		# # s_hard_data = np.array(s_hard_data)
		# # s_hard_target_indices = np.array(s_hard_target_indices)
		# # s_num_hard_data = len(s_hard_target_indices)

		# # s_rtd_data = np.array(s_rtd_data)
		# # s_rtd_hard_data = np.array(s_rtd_hard_data)
		# # s_rtd_hard_target_indices = np.array(s_rtd_hard_target_indices)
		# # s_num_rtd_hard_data = len(s_rtd_hard_target_indices)

		# # assert(s_num_hard_data == t_num_hard_data)
		# # assert(s_num_rtd_hard_data == t_num_rtd_hard_data)

		# # print("The number of problems whose teaching dimension > 1: {}".format(t_num_hard_data))
		# # print("The number of problems whose recursive teaching dimension = 1: {}".format(num_rtd_data))
		# # print("The number of problems whose teaching dimension > 1 and  recursive teaching dimension = 1: {}".format(t_num_rtd_hard_data))
		
		# attr_equiv_class_mappings = np.array(attr_equiv_class_mappings)

		# datasets = {"t_data" : t_data, "t_target_indices" : t_target_indices, \
		# 			"s_data" : s_data, "s_target_indices" : s_target_indices, \
		# 			"attr_equiv_class" : attr_equiv_class_mappings, \
		# 			"num_data" :  num_of_games, \

		# 			# "t_hard_data" : t_hard_data, "t_hard_target_indices" : t_hard_target_indices, \
		# 			# "s_hard_data" : s_hard_data, "s_hard_target_indices" : s_hard_target_indices, \
		# 			# "num_hard_data" : t_num_hard_data, \

		# 			# "t_rtd_data" : t_rtd_data, "t_rtd_target_indices" : t_rtd_target_indices, \
		# 			# "s_rtd_data" : s_rtd_data, "s_rtd_target_indices" : s_rtd_target_indices, \
		# 			# "num_rtd_data" : num_rtd_data, \

		# 			# "t_rtd_hard_data" : t_rtd_hard_data, "t_rtd_hard_target_indices" : t_rtd_hard_target_indices, \
		# 			# "s_rtd_hard_data" : s_rtd_hard_data, "s_rtd_hard_target_indices" : s_rtd_hard_target_indices, \
		# 			# "num_rtd_hard_data" : t_num_rtd_hard_data
		# 				}
		# # with open("dataset.p", 'wb') as fp:
		# # 	pickle.dump(datasets, fp, protocol=pickle.HIGHEST_PROTOCOL)
		# # with open('dataset.p', 'rb') as fp:
		# # 	data = pickle.load(fp)
		# # 	print(data)

		# return datasets
	
	def generate_batch(self, batch_size, role = None):
		data = {'prev_belief': [], 'message': [], 'distractors': [], 'new_belief': []}
		for i in range(batch_size):
			if i % 8 == 0:
				seed = int(time.time())
			prev_belief = np.random.random(self.num_distractors)
			prev_belief /= np.sum(prev_belief)
			discrete_concepts, included_attributes, embeded_concepts, _, _, _, _ = self.rd_generate_concept()
			#msg = included_attributes[np.random.randint(len(included_attributes))]
			msg = self.attributes[np.random.randint(len(self.attributes))]
			embeded_msg = np.zeros(len(self.attributes))
			embeded_msg[msg] = 1
			new_belief = self.bayesian_update(prev_belief, discrete_concepts[0], msg)
			data['prev_belief'].append(prev_belief)
			data['message'].append(embeded_msg)
			data['distractors'].append(embeded_concepts)
			data['new_belief'].append(new_belief)
		for k in data:
			data[k] = np.array(data[k])
		return data


def main():
	import time
	from pprint import pprint as brint
	game_config = edict()
	game_config.num_distractors = 7
	game_config.attributes = range(12)
	game_config.concept_max_size = 5
	game_config.permutation_train = True
	game_config.generated_dataset = False
	game_config.generated_dataset_path_train = '../../../../Datasets/Number_Set/Number_Set_%ddis_Train_2.npy' % game_config.num_distractors
	concept_space = Concept(game_config, True)
	concept_space.generate_novel_dataset(0.3, 400000, 100000)
	# train = concept_space.generate_training_dataset(600000)
	# test = concept_space.generate_testing_dataset(400000)
	# t1 = time.time()
	# for _ in range(100000):
	# 	concept_space.rd_generate_concept()
		# input()
	# print(time.time() - t1)
	# exit()
	
	#concept_space = Concept(game_config)
	#test = concept_space.generate_testing_dataset(20000)
	# train = np.load('/home/Datasets/Number_Set_Train.npy').item()
	# test = np.load('/home/Datasets/Number_Set_Test.npy').item()
	# print(train)
	# print(len(train['discrete_concepts']))
	# print(test)
	# print(len(test['discrete_concepts']))
	'''
	print(included_attributes)
	belief = np.ones(num_concepts) / num_concepts
	print('old belief:', belief)
	msg = np.random.randint(10)
	belief = concept_space.bayesian_update(belief, concepts, msg)
	print('message is', msg)
	print('new belief:', belief)

	number_sets = [[1, 2, 3, 4],
				   [3, 4, 5, 6],
				   [2, 4, 5, 7],
				   [2, 3, 5, 8],
				   [2, 3, 4, 5]]
	belief1 = concept_space.bayesian_update(np.ones(5) / 5, number_sets, 2)
	belief2 = concept_space.bayesian_update(belief1, number_sets, 3)
	belief3 = concept_space.bayesian_update(belief2, number_sets, 4)
	belief4 = concept_space.bayesian_update(belief3, number_sets, 5)

	objects = ['{' + ' '.join(str(e) for e in i) % i + '}' for i in number_sets]
	y_pos = np.arange(len(objects))

	# plt.bar(y_pos, belief4, align='center', alpha=0.5)
	# plt.ylim(top = 1)  # adjust the top leaving bottom unchanged
	# plt.ylim(bottom = 0)
	# plt.xticks(y_pos, objects)
	# plt.ylabel('States')
	# plt.title('Belief given message 2')
	#plt.show()
	num_concepts = 4
	concept_size = 4
	concept_space = Concept(attributes, num_concepts, concept_size)
	rtd_large = 0
	num_samples = 10000
	count = 0
	for i in range(10000):
		concepts, included_attributes, concept_embeddings, _ = concept_space.rd_generate_concept()
		if concept_space.teaching_dim(concepts, included_attributes):
			count += 1
	print(count)
	input()
	for _ in tqdm(range(num_samples)):
		concepts, included_attributes, concept_embeddings, _ = concept_space.rd_generate_concept()
		combinations = get_combination(list(range(4)), 2)
		#print(concepts)
		td_dict = concept_space.teaching_dim(concepts, included_attributes)
		#print(td_dict)
		rtd = concept_space.recursive_teaching_dim(concepts)
		#print('rtd is : %d' % rtd)
		if rtd > 1:
			rtd_large += 1
	print(rtd_large * 1.0 / num_samples)
	'''

if __name__ == '__main__':
	main()
