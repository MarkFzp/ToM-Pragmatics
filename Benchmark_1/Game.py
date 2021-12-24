from Speaker import Speaker
from Listener import Listener
import tensorflow as tf
import numpy as np
import os, sys
from easydict import EasyDict as edict
from tensorflow.python import debug as tf_debug

'''
Naming Convention:
    plural: lists, 1d numpy arrays
    singular: primitive types, tensors, 2d numpy arrays, ...

Placeholders:
    self.speaker_.target
    self.speaker_.is_train
    self.listener_.data
    self.listener_.target_index
    self.listener_.is_train
'''

class Game:
    '''
    data: numpy 2d array for FcEncoder / numpy 4d NHWC for CnnEncoder
    target_indices: [numpy 1d array] eg. [3, 1, 4, 2, 4] when candidate_size is 5, batch_size is 5
    message_len: max message length
    '''
    def __init__(self, sess, encoder_type,  input_data_len, batch_size, num_epoches, max_concept_len,  num_candidate, num_attributes, dense_len, message_len, alphabet_size, temperature=1.5, \
                 beta_speaker=0.01, beta_listener=0.001, learning_rate=0.001, separate_optimize=False, concept_folder = None, train_permutation=False, test_permutation=False, train_set_path=None, test_set_path=None, **kwargs):
        assert(encoder_type in ['fc', 'cnn'])
        self.sess = sess
        self.encoder_type_ = encoder_type
        self.batch_size_ = batch_size
        self.num_candidates_ = num_candidate
        self.num_epoches_ = num_epoches
        self.input_data_len_ = input_data_len
        self.concept_folder_ = concept_folder
        self.train_permute_ = train_permutation
        self.test_permute_ = test_permutation
        self.train_set_path_ = train_set_path
        self.test_set_path_ = test_set_path
        if self.encoder_type_ == 'fc':
            self.num_attributes = num_attributes
        elif self.encoder_type_ == 'cnn':
            self.num_attributes = None
      
        assert(self.num_candidates_ == int(self.num_candidates_))
       
        self.max_concept_len_ = max_concept_len
        self.dense_len_ = dense_len
        self.message_len_ = message_len
        self.alphabet_size_  = alphabet_size
        self.temperature_ = temperature
        self.beta_speaker_ = beta_speaker
        self.beta_listener_ = beta_listener
        self.learning_rate_ = learning_rate
        self.separate_optimize_ = separate_optimize
        self.global_step_ = tf.get_variable('global_step_', initializer=tf.constant(0), trainable=False)

        self.kwargs_ = dict()
        if encoder_type == 'cnn':
            self.height_ , self.width_, self.channel_ = data.shape[1:]
            keys = list(kwargs)
            assert('strides' in keys)
            self.strides_ = kwargs['strides']
            self.kwargs_ = {'strides': self.strides_, 'height': self.height_, 'width': self.width_, 'channel': self.channel_}

        #########################################
        # structures for pretrain speaker
        #########################################
        self.speaker_ = Speaker(self.encoder_type_, self.input_data_len_, self.dense_len_, self.message_len_, self.alphabet_size_, self.temperature_, **self.kwargs_)
        self.speaker_logits_  = self.speaker_.logits
        # self.pred_attr_ = tf.sigmoid(self.speaker_logits_)
        self.pred_attr_spvs = tf.placeholder(dtype = tf.float32, shape = (None, self.num_attributes))
        self.pretrain_speaker_loss_ = tf.losses.sigmoid_cross_entropy(self.pred_attr_spvs, self.speaker_logits_)
        self.speaker_pretrain_op_ = tf.train.RMSPropOptimizer(learning_rate = 1e-4).minimize(self.pretrain_speaker_loss_)
        self.kernel_0 = tf.get_collection(tf.GraphKeys.VARIABLES, 'Teacher_Update/fcencoder/kernel')        
        #########################################
        # structures for pretrain listener
        #########################################
        self.message_ = self.speaker_.message
        self.listener_ = Listener(self.encoder_type_, self.input_data_len_, self.dense_len_, self.num_candidates_, self.message_, self.message_len_, \
                                  self.alphabet_size_, self.temperature_, **self.kwargs_)

        self.listener_match_indicator = self.listener_.match_indicator
        self.listener_log_prob_ = self.listener_.log_prob
        self.listener_entropy_ = self.listener_.entropy
        self.pretrain_listener_reward = self.listener_match_indicator * (self.listener_log_prob_ + self.beta_listener_ * self.listener_entropy_)
        self.pretrain_listener_loss = -tf.reduce_mean(self.pretrain_listener_reward)
        self.listener_pretrain_op_ = tf.train.RMSPropOptimizer(learning_rate = 1e-4).minimize(self.pretrain_listener_loss)
        self.pretrain_listener_accr = tf.reduce_mean(self.listener_match_indicator)
        ########################################
        # structure for training
        ########################################

        self.speaker_log_prob_sum_ = self.speaker_.log_prob_sum
        self.speaker_entropy_ = self.speaker_.entropy
        self.cur_accr_ = tf.reduce_mean(self.listener_match_indicator)

        if not separate_optimize:
            self.reward = self.listener_match_indicator * (self.speaker_log_prob_sum_ + self.listener_log_prob_ + self.beta_speaker_ * self.speaker_entropy_ + self.beta_listener_ * self.listener_entropy_)
            # self.reward = 2 * (self.listener_match_indicator - 0.5) * (self.speaker_log_prob_sum_ + self.listener_log_prob_ + self.beta_speaker_ * self.speaker_entropy_ + self.beta_listener_ * self.listener_entropy_)
            self.loss = -tf.reduce_mean(self.reward)
            self.optimize = tf.train.RMSPropOptimizer(self.learning_rate_).minimize(self.loss, global_step=self.global_step_)
        else:
            self.speaker_reward = self.listener_match_indicator * (self.speaker_log_prob_sum_ + self.beta_speaker_ * self.speaker_entropy_)
            self.speaker_loss = -tf.reduce_mean(self.speaker_reward)
            self.listener_reward = self.listener_match_indicator * (self.listener_log_prob_ + self.beta_listener_ * self.listener_entropy_)
            self.listener_loss = -tf.reduce_mean(self.listener_reward)
            self.loss = self.speaker_loss + self.listener_loss
            self.speaker_optimize = tf.train.RMSPropOptimizer(self.learning_rate_).minimize(self.speaker_loss, global_step=self.global_step_)
            self.listener_optimize = tf.train.RMSPropOptimizer(self.learning_rate_).minimize(self.listener_loss)
            # name_list = ['dense_1/kernel:0', 'dense_1/bias:0', 'LstmM2D/rnn/lstm_cell/kernel:0', 'LstmM2D/rnn/lstm_cell/bias:0']
            # self.listener_optimize = tf.train.RMSPropOptimizer(self.learning_rate_).minimize(-self.listener_reward, var_list=[v for v in tf.global_variables() if v.name in name_list])

    '''
    return: [tuple] (targets, data, target_indices)
    '''
    # def read_batch(self, batch_size, random=True, one_round=False):
    #     while True:
    #         indices = np.arange(self.target_size_, dtype=np.int32)
    #         if random:
    #             np.random.shuffle(indices)
    #         for i in range(self.target_size_ // batch_size):
    #             batch_indices = indices[np.arange(i * batch_size, (i + 1) * batch_size)]
    #             batch_targets = self.target_[batch_indices]
    #             batch_data = self.data_[np.array([np.arange(i * self.num_candidates_, (i + 1) * self.num_candidates_) for i in batch_indices]).reshape((-1, ))]
    #             batch_target_indices = self.target_indices_[batch_indices]
    #             yield (batch_targets, batch_data, batch_target_indices)
    #         if one_round:
    #             break

    # def generate_batch(self, batch_size):
    #     import sys
    #     sys.path.insert(1, '../CL_Net/Referential_Game/Number_Set')
    #     from concept import Concept
    #     game_config = edict({'attributes': range(self.num_attributes), 'num_distractors': self.num_candidiate, 'concept_max_size' : 5 })
        
    #     concept_space = Concept(game_config)
    #     while True:
    #         # print("inside!")
    #         batch_data = []
    #         for _ in range(batch_size):
    #             for concept in concept_space.rd_generate_concept()[2]:
    #                 batch_data.append(concept)
    #         batch_data = np.array(batch_data)
    #         batch_target_indices = np.random.choice(range(self.num_candidiate), batch_size)
    #         batch_targets = batch_data[batch_target_indices + np.arange(0, batch_size * self.num_candidiate, self.num_candidiate)]
    #         yield (batch_targets, batch_data, batch_target_indices)
    def gen_batch_from_datasets(self):
        print("---------------------Start generate data from dataset------------------------")

        #load the dataset
        sys.path.append('../CL_Net/Referential_Game/' + self.concept_folder_)

        from concept import Concept
        from easydict import EasyDict as edict

        game_config = edict({'attributes': list(range(self.num_attributes)), 'num_distractors': self.num_candidates_, \
            'concept_max_size' : self.max_concept_len_, 'permutation' : self.train_permute_, \
            'generated_dataset' : True, 'generated_dataset_path' : self.dataset_path_})    
        concept_space = Concept(game_config)
        _, _, t_data, s_data, \
		attr_equiv_class_mappings, t_s_concept_idx_mappings, t_target_indices = concept_space.rd_generate_concept()        
        
        num_games = t_target_indices.shape[0]
        #generate teacher and student's indices
        t_target_indices = np.array(t_target_indices)
        t_s_concept_idx_mappings = np.array(t_s_concept_idx_mappings)
        s_target_indices = np.where(t_s_concept_idx_mappings == t_target_indices.reshape(-1,1))[1]
        t_targets = t_data[t_target_indices + np.arange(0, self.batch_size_ * self.num_candidates_, self.num_candidates_)] 

        for epoch in range(self.num_epoches_):
            game_indices = np.arange(num_games, dtype=np.int32)
            np.random.shuffle(game_indices)

            num_batch_per_epoch = num_games // self.batch_size_
            if num_games % self.batch_size_ != 0:
                num_batch_per_epoch += 1
            
            batch = {}
            for i in range(num_batch_per_epoch):
                batch_indices = game_indices[np.arange(i * self.batch_size_, min((i + 1) * self.batch_size_, num_games))]
                true_batch_size = self.batch_size_
                if (i+1) * self.batch_size_ > num_games:
                    true_batch_size = num_games - i * self.batch_size_
                t_batch_targets = t_targets[batch_indices]
                #batch_data has shape batch_size * num_candidates * input_len
                
                # t_batch_data = t_data[np.array([np.arange(i * self.num_candidates_, (i + 1) * self.num_candidates_) for i in batch_indices]).flatten()]
                s_batch_data = s_data[np.array([np.arange(i * self.num_candidates_, (i + 1) * self.num_candidates_) for i in batch_indices]).flatten()]
                t_batch_target_indices = t_target_indices[batch_indices]
                s_batch_target_indices = s_target_indices[batch_indices]
                batch_attr_equiv_class_mapping = attr_equiv_class_mappings[batch_indices]
                mask = np.ones((true_batch_size, self.num_candidates_), dtype=bool)
                mask[np.arange(true_batch_size), t_batch_target_indices] = False
                # t_batch_distractors = t_batch_data[mask].reshape(true_batch_size, self.num_candidates_-1, self.num_attributes)

                batch['t_targets'] = t_batch_targets
                # batch['t_distractors'] = t_batch_distractors
                # batch['t_target_indices'] = t_target_indices
                batch['s_data'] = s_batch_data
                batch['s_target_indices'] = s_batch_target_indices
                batch['attributes'] = t_batch_targets
                batch['attr_equiv_class'] = batch_attr_equiv_class_mapping

                yield batch


    def rd_gen_batch(self, is_pretrain = False, gen_from_dataset = False):
        print("---------------------Start random generate data------------------------")
        
        sys.path.append('../CL_Net/Referential_Game/' + self.concept_folder_)
        from concept import Concept

        train_permute = self.train_permute_
        if is_pretrain:
            train_permute = False

        generated_dataset = False
        if gen_from_dataset:
            generated_dataset = True

        game_config = edict({'attributes': range(self.num_attributes), 'num_distractors': self.num_candidates_, \
                    'concept_max_size' : self.max_concept_len_, 'permutation_train' : train_permute,"permutation_test": self.test_permute_,\
                    'generated_dataset' : generated_dataset, 'generated_dataset_path_train' : self.train_set_path_, 'generated_dataset_path_test':self.test_set_path_,\
                    'dataset_attributes_path' : self.dataset_attributes_path_,\
                    'feat_dir' : self.pretrain_feature_path_,\
                    'img_h' : 128, 'img_w' : 128, 'save_dir' : None} )    


        # game_config = edict({'attributes': range(self.num_attributes_), 'num_distractors': self.num_candidates_, 'concept_max_size' : 5 })    
        concept_space = Concept(game_config, True)
        # concept_space = Concept(range(self.num_attributes_), self.num_candidates_, 4)
        batch = {}

        while True:
            t_data = []
            s_data = []
            attr_equiv_class_mappings = []
            t_s_concept_idx_mappings = []
            t_target_indices = []
            t_attr_spvs = []

            for _ in range(self.batch_size_):
                (t_concepts,_), _,\
			    t_concept_embed, s_concept_embed, \
			    attr_equiv_class_mapping, t_s_concept_idx_mapping, t_target_idx = concept_space.rd_generate_concept()

                t_data.append(t_concept_embed)
                s_data.append(s_concept_embed)
                attr_equiv_class_mappings.append(attr_equiv_class_mapping)
                
                t_s_concept_idx_mappings.append(t_s_concept_idx_mapping)
                t_target_indices.append(t_target_idx)
            

                #generate concept embed for pretrain supervision
                spvs_temp = np.zeros(self.num_attributes)
               
                spvs_temp[np.array(t_concepts[t_target_idx])] = 1
                t_attr_spvs.append(spvs_temp)


            #generate teacher's data
            t_target_indices = np.array(t_target_indices)
            t_data = np.array(t_data).reshape(-1, self.input_data_len_)
            s_data = np.array(s_data).reshape(-1, self.input_data_len_)
            t_attr_spvs = np.array(t_attr_spvs)

            # print(t_target_indices + np.arange(0, self.batch_size_ , self.num_candidates_))
            # print(t_data.shape)
            t_targets = t_data[t_target_indices + np.arange(0, self.batch_size_ * self.num_candidates_, self.num_candidates_)] 
            # mask = np.ones((self.batch_size_, self.num_candidates_), dtype=bool)
            # mask[np.arange(self.batch_size_), t_target_indices] = False
            # t_data = t_data.reshape(self.batch_size_, self.num_candidates_, self.num_attributes)
            # t_distractors = t_data[mask].reshape(self.batch_size_, self.num_candidates_-1, self.num_attributes)
            t_attributes = t_targets
            
            #generate student's data
            s_data = np.array(s_data)
            t_s_concept_idx_mappings = np.array(t_s_concept_idx_mappings)
            s_target_indices = np.where(t_s_concept_idx_mappings == t_target_indices.reshape(-1,1))[1]

            #     for concept in concept_space.rd_generate_concept()[2]:
            #         batch_data.append(concept)
            # batch_data = np.array(batch_data)
            # batch_target_indices = np.random.choice(range(self.num_candidates_), self.batch_size_)

            attr_equiv_class_mappings = np.array(attr_equiv_class_mappings)

            # datasets = concept_space.generate_fixed_datasets(self.batch_size_, equiv = self.permutation_)
            # t_batch_data = datasets["t_data"]
            # t_target_indices = datasets["t_target_indices"]
            # t_batch_targets = t_batch_data[t_target_indices + np.arange(0, self.batch_size_ * self.num_candidates_, self.num_candidates_)]
            # mask = np.ones((self.batch_size_, self.num_candidates_), dtype=bool)
            # mask[np.arange(self.batch_size_), t_target_indices] = False
            # t_batch_data = t_batch_data.reshape(self.batch_size_, self.num_candidates_, self.num_attributes_)
            # t_batch_distractors = t_batch_data[mask].reshape(self.batch_size_, self.num_candidates_-1, self.num_attributes_)
            # t_batch_attributes = t_batch_targets
            
            # s_batch_data = datasets["s_data"]
            # s_target_indices = datasets["s_target_indices"]

            batch['t_targets'] = t_targets
            # batch['t_distractors'] = t_distractors
            batch['t_target_indices'] = t_target_indices
            batch['s_data'] = s_data
            batch['s_target_indices'] = s_target_indices
            batch['attributes'] = t_attr_spvs
            batch['attr_equiv_class'] = attr_equiv_class_mappings

            yield batch

    def evaluate(self, num_iter, ckpt_folder_name, ckpt_name,  dataset_attributes_path = None, pretrain_feature_path = None):
        
        #load checkpoint
        loader = tf.train.Saver()
        appendix = None
        ckpt_folder_name = "./" + ckpt_folder_name + "/"
        for file in os.listdir(ckpt_folder_name):
            if ckpt_name in file:
                try:
                    # print(file.replace(ckpt_name + '-', '').split('.')[0])
                    file_appendix = int(file.replace(ckpt_name + '-', '').split('.')[0])
                except ValueError:
                    continue
                if appendix is None or file_appendix > appendix:
                    appendix = file_appendix
        if appendix is not None:
            latest_filename = ckpt_name + '-' + str(appendix)
            # ckpt = tf.train.get_checkpoint_state(ckpt_folder_name, latest_filename=latest_filename)
            # print(ckpt)
            # if ckpt and ckpt.model_checkpoint_path:
            loader.restore(self.sess, ckpt_folder_name + latest_filename)
            print('Successfully load checkpoint ...')
            print("checkpoint path: {}".format(ckpt_folder_name + latest_filename))
            # total_iter = self.sess.run(self.global_step_)
        
        #load the test dataset
        sys.path.append('../CL_Net/Referential_Game/' + self.concept_folder_)
        from concept import Concept

        game_config = edict({'attributes': range(self.num_attributes), 'num_distractors': self.num_candidates_, \
        'concept_max_size' : self.max_concept_len_,'permutation_train':self.train_permute_, 'permutation_test' : self.test_permute_, \
        'generated_dataset' : True,'generated_dataset_path_train': self.train_set_path_, 'generated_dataset_path_test' : self.test_set_path_, \
        'dataset_attributes_path' : dataset_attributes_path,\
        'feat_dir' : pretrain_feature_path,\
        'img_h' : 128, 'img_w' : 128, 'save_dir' : None} )    

       
        concept_space = Concept(game_config,False)
        test_accr = []
        rtd_accr = []
        hard_accr = []
        rtd_hard_accr = []  
        game_mean_coss = []

        for _ in range(num_iter):
            (t_concepts, s_concepts), (t_included_attr, s_included_attr), t_data, s_data, \
            attr_equiv_class_mapping, t_s_concept_idx_mapping, t_target_idx = concept_space.rd_generate_concept()
            # num_games = len(t_target_indices) 
            # print("total number of test game: {}".format(num_games))

            #generate teacher and student's indices
            # t_target_indices = np.array(t_target_indices)
            # t_s_concept_idx_mappings = np.array(t_s_concept_idx_mappings)
            s_target_idx = np.where(t_s_concept_idx_mapping == t_target_idx)[0]
            t_target = t_data[t_target_idx]
            # num_test_data = t_target_idx.shape[0]

            #generate distractors
            mask = np.ones((self.num_candidates_), dtype = bool)
            mask[ t_target_idx] = False
            # t_distractors = t_data[mask].reshape(-1, self.num_candidates_-1, self.input_data_len_)
            
            t_cosine = np.sum(t_data * t_target, axis = 1) / (np.sqrt(np.sum(t_data ** 2, axis = 1)) * np.sqrt(np.sum(t_target ** 2)))
            s_cosine = np.sum(s_data * s_data[s_target_idx], axis = 1) / (np.sqrt(np.sum(s_data ** 2, axis = 1)) * np.sqrt(np.sum(s_data[s_target_idx] ** 2)))
            
        
            # t_cosine_max = t_cosine[np.argsort(t_cosine)[-2]]
            # s_cosine_max = s_cosine[np.argsort(s_cosine)[-2]]
            # game_max_cos = np.mean([t_cosine_max,s_cosine_max])
            game_mean_cosine = np.mean(s_cosine[np.argsort(s_cosine)[:-1]])
            assert(game_mean_cosine <=1)
            # game_max_coss.append(game_max_cos)
            game_mean_coss.append(game_mean_cosine)
            t_target = t_target.reshape(-1, self.input_data_len_)

            fd = {self.speaker_.target: t_target, \
                self.speaker_.is_train : False,\
                    self.listener_.data: s_data, \
                    self.listener_.target_index: s_target_idx, \
                    self.listener_.attr_equiv_class_: attr_equiv_class_mapping.reshape(-1, self.num_attributes), \
                    self.listener_.is_train : False}
            accr = self.sess.run([self.cur_accr_], feed_dict = fd)
            test_accr.append(accr)


            t_td_dict, _ = concept_space.teaching_dim(t_concepts, t_included_attr)
            s_td_dict, _ = concept_space.teaching_dim(s_concepts, s_included_attr)
            #append rtd=1 accuracy
            if concept_space.recursive_teaching_dim(t_concepts) == 1:
                rtd_accr.append(accr)
                # print(t_td_dict[t_concepts[t_target_idx]])
                if t_td_dict[t_concepts[t_target_idx]][0] > 1:
                    rtd_hard_accr.append(accr)

            if t_td_dict[t_concepts[t_target_idx]][0] > 1:
                hard_accr.append(accr)
            test_accr.append(accr)

        print("test accuracy: {}".format(np.mean(test_accr)))
        # print("rtd = 1 accuracy: {}".format(np.mean(rtd_accr)))
        # print("rtd = 1 hard accuracy: {}".format(np.mean(rtd_hard_accr)))
        # print("hard accuracy: {}".format(np.mean(hard_accr)))
        # evaluation_dict = { "game_mean_coss" : game_mean_coss, "test_accr" : test_accr,\
        #     "rtd_accr" : rtd_accr, "rtd_hard_accr" : rtd_hard_accr, "hard_accr" : hard_accr}
        # np.save(ckpt_folder_name + "cos_dict.npy", evaluation_dict)

        # game_mean_coss = np.array(game_mean_coss)
        test_accr = np.array(test_accr)
        top_ten_idx = np.argsort(game_mean_coss)[-int(len(game_mean_coss)/10):]
        print("top 10% test accuracy: {}".format(np.mean(test_accr[top_ten_idx])))
        # pt_9_idx = np.where(game_mean_coss >= 0.9)[0]
        # pt_8_idx = np.where(game_mean_coss >= 0.8)[0]
        # pt_7_idx = np.where(game_mean_coss >= 0.7)[0]
        # pt_6_idx = np.where(game_mean_coss >= 0.6)[0]
        # pt_5_idx = np.where(game_mean_coss >= 0.5)[0]
        # pt_4_idx = np.where(game_mean_coss >= 0.4)[0]
        # pt_3_idx = np.where(game_mean_coss >= 0.3)[0]
        # pt_2_idx = np.where(game_mean_coss >= 0.2)[0]
        # pt_1_idx = np.where(game_mean_coss >= 0.1)[0]

        # print("accuracy of cos > 0.9: {}".format(np.mean(test_accr[pt_9_idx])))
        # print("accuracy of cos > 0.8: {}".format(np.mean(test_accr[pt_8_idx])))
        # print("accuracy of cos > 0.7: {}".format(np.mean(test_accr[pt_7_idx])))
        # print("accuracy of cos > 0.6: {}".format(np.mean(test_accr[pt_6_idx])))
        # print("accuracy of cos > 0.5: {}".format(np.mean(test_accr[pt_5_idx])))
        # print("accuracy of cos > 0.4: {}".format(np.mean(test_accr[pt_4_idx])))
        # print("accuracy of cos > 0.3: {}".format(np.mean(test_accr[pt_3_idx])))
        # print("accuracy of cos > 0.2: {}".format(np.mean(test_accr[pt_2_idx])))
        # print("accuracy of cos > 0.1: {}".format(np.mean(test_accr[pt_1_idx])))
        
      
        return



    def train(self, max_pretrain_iteration, ckpt_name, folder_name, pretrain = True, dataset_attributes_path = None, pretrain_feature_path = None):
      
        self.pretrain_ = pretrain
        
        self.dataset_attributes_path_ = dataset_attributes_path
        self.pretrain_feature_path_ = pretrain_feature_path
       
        self.sess.run(tf.global_variables_initializer())
        pretrain_speaker_saver = tf.train.Saver()
        pretrain_listener_saver = tf.train.Saver()
        train_saver = tf.train.Saver(max_to_keep = 100)
        loader = tf.train.Saver()

        self.batch_generator_ = self.rd_gen_batch
        
        train_folder_name = "./" + folder_name +"/"
        # if self.train_permute_:
        #         if self.pretrain_:
        #             train_folder_name = "./checkpoints_train_" + dataset_name + "_with_permute_with_pretrain/"
        #         else:
        #             train_folder_name = "./checkpoints_train_" + dataset_name + "_with_permute_no_pretrain/"

        # else:
        #     if self.pretrain_:
        #         train_folder_name = "./checkpoints_train_" + dataset_name + "_without_permute_with_pretrain/"
        #     else:
        #         train_folder_name = "./checkpoints_train_" + dataset_name + "_without_permute_no_pretrain/" 

        if self.pretrain_:
            # if self.train_permute_:
            #     pretrain_listener_folder_name = "./checkpoints_pretrain_" + dataset_name + "_listener_with_permute/"
            #     pretrain_speaker_folder_name = "./checkpoints_pretrain_" + dataset_name + "_speaker_with_permute/"
            # else:
            #     pretrain_listener_folder_name = "./checkpoints_pretrain_" + dataset_name + "_listener_without_permute/"
            #     pretrain_speaker_folder_name = "./checkpoints_pretrain_" +  dataset_name + "_speaker_without_permute/"
            pretrain_speaker_folder_name = "./pretrain_" + folder_name + "_speaker/"
            pretrain_listener_folder_name = "./pretrain_" + folder_name + "_listener/"

            pretrain_speaker_file_name = pretrain_speaker_folder_name + ckpt_name
            pretrain_listener_file_name = pretrain_listener_folder_name + ckpt_name


            if not os.path.exists(pretrain_speaker_folder_name):
                os.makedirs(pretrain_speaker_folder_name)
            if not os.path.exists(pretrain_listener_folder_name):
                os.makedirs(pretrain_listener_folder_name)

        
            cur_iter = 0
            #pretrain speaker
            try:
                loader.restore(self.sess, pretrain_speaker_file_name + '-' + str(max_pretrain_iteration))
            except ValueError: 
                print("Fail to restore checkpoint, start prtraining speaker...")
                for batch in self.batch_generator_(is_pretrain = True):
                    fd = {self.speaker_.target: batch["t_targets"], \
                        self.speaker_.is_train : True, \
                        self.pred_attr_spvs: batch["attributes"]}
                    _, pretrain_speaker_loss = self.sess.run([self.speaker_pretrain_op_, self.pretrain_speaker_loss_], feed_dict = fd)
                   

                    if  cur_iter % 5000 == 0:
                        print("iter {} pretrain speaker loss: {}".format(cur_iter,pretrain_speaker_loss))
                        pretrain_speaker_saver.save(self.sess, pretrain_speaker_file_name + '-' + str(cur_iter))
            
                    cur_iter += 1
                    if cur_iter > max_pretrain_iteration:
                        break
            else:
                print("Successfully loaded checkpoint: {}".format(pretrain_speaker_file_name))

            #pretrain listener
            cur_iter = 0
            try:
                loader.restore(self.sess, pretrain_listener_file_name + '-' + str(max_pretrain_iteration))
            except ValueError: 
                print("Fail to restore checkpoint, start prtraining listener...")
                accrs = []
                for batch in self.batch_generator_(is_pretrain = True):
                    fd = {self.speaker_.target: batch["t_targets"], \
                        self.speaker_.is_train : True, \
                        self.listener_.data: batch["s_data"],
                        self.listener_.target_index : batch["s_target_indices"],\
                        self.listener_.attr_equiv_class_ : batch["attr_equiv_class"],\
                        self.listener_.is_train: True}
                    
                    # weights = self.sess.run([self.kernel_0], feed_dict = fd)
                    # print("weights before pretrain listener: {}".format(weights))
                    _, pretrain_listener_loss, cur_accr= self.sess.run([self.listener_pretrain_op_, self.pretrain_listener_loss, self.pretrain_listener_accr], feed_dict = fd)
                    accrs.append(cur_accr)
                    # print("weights after pretrain: {}".format(wt_after))

                    # if cur_iter == 2:
                    #     exit()
                    if cur_iter % 5000 == 0:
                        print("iter {} pretrain listener loss: {}".format(cur_iter, pretrain_listener_loss))
                        print("mean accuracy: {}".format(np.mean(accrs)))
                        accrs = []
                        pretrain_listener_saver.save(self.sess, pretrain_listener_file_name + '-' + str(cur_iter))



                    cur_iter += 1
                    if cur_iter > max_pretrain_iteration:
                        break
            else:
                print("Successfully loaded checkpoint: {}".format(pretrain_listener_file_name))

       

        
        if not os.path.exists(train_folder_name):
            os.makedirs(train_folder_name)

        # variables_names = [v.name for v in tf.trainable_variables()]
        # if debug_mode:
        #     # # add_tensor_filter for tf debugger
        #     self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        #     self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        #     # store current variable values to see change in values for each optimization iteration later
        #     variables_old_values = self.sess.run(variables_names)
        
      
        #training
        accrs = []
        current_iter = resume_iter = 0

        latest_ckpt_name = tf.train.latest_checkpoint(train_folder_name)
        if latest_ckpt_name is not None:
            try:
                loader.restore(self.sess, latest_ckpt_name)
            except ValueError: 
                print("Fail to load checkpoint {}".format(latest_ckpt_name))
                
            else:
                print("Successfully loaded checkpoint: {}".format(latest_ckpt_name))
                resume_iter = int(latest_ckpt_name.split('-')[1])
        else:
            print("No training checkpoints to be loaded, training from start")

        for batch in self.batch_generator_(gen_from_dataset = True):
            fd = {self.speaker_.target: batch["t_targets"], \
                  self.speaker_.is_train: True, \
                  self.listener_.data: batch["s_data"], \
                  self.listener_.target_index: batch["s_target_indices"], \
                  self.listener_.attr_equiv_class_ : batch["attr_equiv_class"],\
                  self.listener_.is_train: True}

            # if not self.separate_optimize_:
            _,loss, cur_accr = self.sess.run([self.optimize,self.loss, self.cur_accr_], feed_dict=fd)
            accrs.append(cur_accr)
            
            # else:
            #     self.sess.run([self.speaker_optimize, self.listener_optimize], feed_dict=fd)
            # loss, error, dense, prob, message, h, out_prob, match_indicator = self.sess.run([self.loss, self.error, self.speaker_.dense_, self.speaker_.d2m_.prob, self.speaker_.message, self.speaker_.d2m_.h, self.speaker_.d2m_.out_prob, self.listener_match_indicator], feed_dict=fd)
            # loss, error = self.sess.run([self.loss, self.error], feed_dict=fd)


            # if debug_mode:
            #     print('message: ', message)
            #     print('error: ', error)
            #     print('loss: ', loss)
            #     print('match_indicator: ', match_indicator)
            #     values = self.sess.run(variables_names)
            #     for j, kv in enumerate(zip(variables_names, values)):
            #         k, v = kv
            #         print("Variable: ", k)
            #         print("Shape: ", v.shape)
            #         print("Diff: ", v - variables_old_values[j])
            #         variables_old_values[j] = v
            #     input("Press keyborad to continue next batch ...")

         
            # if current_iter % 1000 == 1:
            #     message, dense = self.sess.run([self.speaker_.message, self.speaker_.dense_], feed_dict=fd)
                # print('dense: ', dense[:10, :])
                # print('targets: ', batch_targets[:10, :])
                # print('error: ', error)
                # print('h: ', h[:10, :])
                # print('out_prob: ', out_prob[:10, :])
                # print('prob: ', prob[:10, :])
                # print('message: ', np.reshape(message, [-1]))
                # values = self.sess.run(variables_names)
                # for k, v in zip(variables_names, values):
                #     print("Variable: ", k)
                #     print("Shape: ", v.shape)
                #     print("Value: ", v)
                # for i, char_list in enumerate(message.T):
                #     unique, counts = np.unique(char_list, return_counts=True)
                #     print('{}th char: '.format(i), dict(zip(unique, counts)))
                # if debug_mode:
                #     input("Press keyborad to continue ...")
                
            if current_iter % 5000 == 0:
                print("iter: {}, loss: {}, cur_accr: {}".format(current_iter, loss, cur_accr))
                print("mean accuracy: {}".format(np.mean(accrs)))
                accrs = []
                train_saver.save(self.sess, train_folder_name + ckpt_name, global_step=resume_iter)

            # take argmax instead of gibbs sampling to get error rate
            if current_iter == 100000:
                return
            
            resume_iter += 1
            current_iter += 1
              
