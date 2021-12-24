import numpy as np
from collections import defaultdict
from tqdm import tqdm
from pprint import pprint as brint
import pdb

class Concept:
	def __init__(self, num_slots):
		self.num_slots_ = num_slots
		self.num_calendars_ = 2 ** self.num_slots_
		binaries = [list(('{0:0%db}' % self.num_slots_).format(i)) for i in range(self.num_calendars_)]
		numbers = []
		for b in binaries:
			numbers.append([int(bb) for bb in b])
		self.tensor_ = np.array(numbers)
		self.tensor_3d_ = np.expand_dims(self.tensor_, 1)
		self.calendar2msg_ = {}
		self.calendar_num_valid_msg_ = np.ones(self.num_calendars_)
		self.all_msgs_, self.all_msgs_tensor_ = self.get_msg(self.num_calendars_ - 1)
		self.num_msgs_ = len(self.all_msgs_)
		#initialize caches about msgs
		for i in range(self.num_calendars_):
			self.get_msg(i)

	def get_msg(self, idx):
		if idx == 0:
			self.calendar2msg_[idx] = ([], np.zeros((1, self.num_slots_)))
		if self.calendar2msg_.get(idx) is None:
			basic_msgs = list(np.nonzero(self.tensor_[idx, :])[0])
			msgs = []
			for first in range(len(basic_msgs)):
				second = first + 1
				while second < len(basic_msgs):
					if basic_msgs[second] == basic_msgs[second - 1] + 1:
						msgs.append(tuple(range(basic_msgs[first], basic_msgs[second] + 1)))
						second += 1
					else:
						break
			msgs += [tuple([m]) for m in basic_msgs]
			msg_tensor = np.zeros((len(msgs), self.num_slots_))
			for im, msg in enumerate(msgs):
				for m in msg:
					msg_tensor[im, m] = 1
			self.calendar2msg_[idx] = (msgs, msg_tensor)
			self.calendar_num_valid_msg_[idx] = len(msgs)

		return self.calendar2msg_[idx]


	def rd_generate_concept(self, without_empty = 0):
		gt_idx = np.random.randint(without_empty, self.num_calendars_)
		msgs, msg_tensor = self.get_msg(gt_idx)
		
		return gt_idx, msgs, msg_tensor

	def bayesian_update(self, old_belief, tensor_info):
		likelihood = (np.sum(self.tensor_ * tensor_info, axis = 1) == np.sum(tensor_info)) / self.calendar_num_valid_msg_
		likelihood *= (np.sum(tensor_info) > 0)
		likelihood[0] = 1 if np.sum(tensor_info) == 0 else 0
		new_belief = old_belief * likelihood
		if np.sum(new_belief) != 0:
			new_belief /= np.sum(new_belief)
		return new_belief

	def generate_batch(self, batch_size):
		data = {'prev_belief': [], 'message': [], 'distractors': [], 'new_belief': []}
		for i in range(batch_size):
			#calender belief
			prev_belief = np.random.random(self.num_calendars_)
			prev_belief /= np.sum(prev_belief)

			cont_msg_num = np.random.randint(1, 4)
			for i in range(cont_msg_num):
				embeded_msg = self.all_msgs_tensor_[np.random.randint(len(self.all_msgs_)), :]
				new_belief = self.bayesian_update(prev_belief, embeded_msg)
				if i < cont_msg_num - 1:
					prev_belief = new_belief.copy()

			#use slot belief instead of calendar belief to train
			prev_belief = np.sum(self.tensor_ * np.expand_dims(prev_belief, 1), axis = 0)
			new_belief = np.sum(self.tensor_ * np.expand_dims(new_belief, 1), axis = 0)
			data['prev_belief'].append(prev_belief)
			data['message'].append(embeded_msg)
			data['distractors'].append(np.expand_dims(self.tensor_, axis = 1))
			data['new_belief'].append(new_belief)
		
		for k in data:
			data[k] = np.array(data[k])
		return data



def main():
	calendars = Concept(3)
	print(calendars.tensor_.shape)
	gt_idx, msgs, msg_tensor = calendars.rd_generate_concept()
	print(calendars.get_msg(0))
	print(calendars.get_msg(3))
	print(calendars.tensor_[gt_idx, ...], msgs, msg_tensor)
	data_batch = calendars.generate_batch(2)
	brint(data_batch)
	print(calendars.bayesian_update(np.ones(2 ** calendars.num_slots_) /\
									(2 ** calendars.num_slots_), np.zeros(calendars.num_slots_)))

	

if __name__ == '__main__':
	main()