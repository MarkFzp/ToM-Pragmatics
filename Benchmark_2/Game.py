from Speaker import Agnostic_Speaker, Informed_Speaker
from Listener import Listener
import tensorflow as tf
import numpy as np
import os
import sys
from easydict import EasyDict as edict
import collections

class Game:
    '''
    data: numpy 2d array for FcEncoder / numpy 4d NHWC for CnnEncoder
    target_indices: [numpy 1d array] eg. [3, 1, 4, 2, 4] when candidate_size is 5, batch_size is 5
    message_len: max message length
    '''
    def __init__(self, sess, encoder_type, speaker_type, input_data_len, batch_size, max_concept_len, num_attributes, num_candidates, dense_len, num_epoches, vocabulary_size=100, num_filters = 20,temperature=10, \
                 learning_rate=0.0001, concept_folder = None, train_permutation=False, test_permutation=False, train_set_path = None, test_set_path =None, **kwargs):
        
        # assert(train_type in ['batch_generation', 'fixed_train_set'])
        assert(encoder_type in ['fc', 'vgg'])
        assert(speaker_type in ['agnostic', 'informed'])
        
        self.sess_ = sess
        # self.train_permutation_ = train_permutation
        self.max_concept_len_ = max_concept_len
        # self.train_type_ = train_type
        self.encoder_type_ = encoder_type
        self.speaker_type_ = speaker_type
        self.num_candidates_ = num_candidates
        self.input_data_len_ = input_data_len
        self.concept_folder_ = concept_folder
        self.train_permutation_ = train_permutation
        self.test_permutation_ = test_permutation
        self.train_dataset_path_ = train_set_path
        self.test_dataset_path_ = test_set_path
        # self.data_size_ = data.shape[0]
        
        

        # if self.train_type_ == 'batch_generation':
        #     self.data_ = None
        #     self.target_indices_ = None
        #     self.target_ = None
        # else:
        #     self.data_ = data
        #     self.target_indices_ = target_indices
        #     self.target_ = data[self.target_indices_ + np.arange(0, self.data_size_, self.num_candidates_, dtype=np.int32)]
             
        if self.encoder_type_ == 'fc':
            self.num_attributes_ = num_attributes
        #if data is image, it should have dimension num_data * img_height * img_width * 3
        elif self.encoder_type_ == 'vgg':
            self.num_attributes_ = None

        # self.num_train_data_ = self.data_size
        
        # self.candidate_count_ = int(self.candidate_count_)
        self.batch_size_ = batch_size
        self.num_epoches_ = num_epoches
        self.dense_len_ = dense_len
        self.vocabulary_size_  = vocabulary_size
        self.temperature_ = temperature
        self.num_filters_ = num_filters
        self.global_step_ = tf.get_variable('global_step_', initializer=tf.constant(0), trainable=False)

        self.learning_rate_ = tf.train.exponential_decay(learning_rate, self.global_step_,100000,0.1, staircase=True)
        
        self.kwargs_ = dict()
    
  
        
    def _build_model(self):

        if self.encoder_type_ == 'vgg':
            pass
            #TODO: to be implemented
            # self.height_ , self.width_, self.channel_ = data.shape[1:]
            # keys = list(kwargs)
            # assert('strides' in keys)
            # self.strides_ = kwargs['strides']
            # self.kwargs_ = {'strides': self.strides_, 'height': self.height_, 'width': self.width_, 'channel': self.channel_}
            
        if self.speaker_type_ == 'agnostic':
            self.speaker_ = Agnostic_Speaker(self.encoder_type_, self.input_data_len_, self.dense_len_, self.num_candidates_-1, self.vocabulary_size_, self.temperature_, sess = self.sess_)
        else:
            self.speaker_ = Informed_Speaker(self.encoder_type_, self.input_data_len_, self.dense_len_, self.num_candidates_-1, self.num_filters_, self.vocabulary_size_, self.temperature_)

        ###############################################
        # structure for pretraining speaker
        ###############################################
        self.speaker_logits_ = self.speaker_.logits
        self.pred_attr_ = tf.sigmoid(self.speaker_logits_)
        self.pred_attr_spvs = tf.placeholder(dtype = tf.float32, shape = (None, self.num_attributes_))
        # self.l2_loss_ = tf.nn.l2_loss(self.pred_attr_spvs - self.pred_attr_)
        self.pretrain_speaker_loss_ = tf.losses.sigmoid_cross_entropy(self.pred_attr_spvs, self.speaker_logits_)
        self.speaker_pretrain_op_ = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(self.pretrain_speaker_loss_)
        self.kernel_0 = tf.get_collection(tf.GraphKeys.VARIABLES, 'Teacher_Update/feature_map/kernel')[0]
        self.kernel_1 = tf.get_collection(tf.GraphKeys.VARIABLES, 'Teacher_Update/combined_feature_map/kernel')

        ###############################################
        # structure for pretraining listener
        ###############################################
        self.speaker_msg_ = self.speaker_.message
        # self.attr_equiv_class_ = tf.convert_to_tensor(attr_equiv_class)
        # self.attr_equiv_class_ = tf.constant(attr_equiv_class)
        # temp_mssage = self.msgs_
        # temp_mssage.set_shape([None])
        # temp = np.zeros((self.msgs_.shape[0],self.msgs_.shape[1] ))
        # self.msgs_ = attr_equiv_class[np.array(self.msgs_, dtype=bool)]
        # self.casted_bool = tf.cast(temp_mssage, dtype = tf.bool)
        # self.bool_shape = tf.shape(self.casted_bool)
        # self.attr_shape = tf.shape(self.attr_equiv_class_)
        # self.mask = tf.boolean_mask(self.attr_equiv_class_, self.casted_bool)
        # self.msgs_ = tf.one_hot( self.mask, self.vocabulary_size_)
        # self.msgs_ = temp[np.arange(self.msgs_.shape[0]), attr_equiv_class[np.array(self.msgs_, dtype=bool)]]

        self.listener_ = Listener(self.input_data_len_, self.dense_len_, self.num_candidates_, self.speaker_msg_, self.vocabulary_size_, self.temperature_)
        self.listener_msg_ = self.listener_.message_
        self.listener_logits_ = self.listener_.logits
        self.listener_log_prob_ = self.listener_.log_prob
        
        self.index_match_indicator_ = self.listener_.index_match_indicator
        self.listener_reward_ =  tf.cast(self.index_match_indicator_, tf.float32) *  self.listener_log_prob_
        self.pretrain_listener_loss_ = tf.reduce_mean(- self.listener_reward_)
        self.pretrain_listener_accr_ = tf.reduce_mean(tf.cast(self.index_match_indicator_, dtype= tf.float64))
        # self.one_hot_target_indices_ = tf.one_hot(self.listener_.target_idx_, depth = self.num_candidates_)
        # self.task_cross_entropy_ = tf.losses.softmax_cross_entropy(self.one_hot_target_indices_, self.listener_logits_)
        self.listener_pretrain_op_ = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(self.pretrain_listener_loss_)

        ###############################################
        # structure for training
        ###############################################
        self.speaker_log_prob_ = self.speaker_.log_prob
        
        self.reward_ = tf.cast(self.index_match_indicator_, tf.float32) * (self.speaker_log_prob_ + self.listener_log_prob_)
        # self.speaker_reg_loss_ = tf.add_n([ tf.nn.l2_loss(v) for v in self.speaker_.reg_varlist_ if 'bias' not in v.name ])
        # self.listener_reg_loss_ = tf.add_n([ tf.nn.l2_loss(v) for v in self.listener_.reg_varlist_ if 'bias' not in v.name ])
        self.weight_loss_ = tf.reduce_mean(- self.reward_)
        self.loss_ =  self.weight_loss_ 
        self.loss_summary_ = tf.summary.scalar("Loss", self.loss_)
        self.train_op_ = tf.train.AdamOptimizer(self.learning_rate_).minimize(self.loss_, global_step=self.global_step_)

        self.curr_accr_ = tf.reduce_mean(tf.cast(self.index_match_indicator_, dtype=tf.float64))
        tf.get_default_graph().add_to_collection("Accuracy", self.curr_accr_)

    # def _build_model(self):
    #     self.global_step_ = tf.get_variable('global_step_', initializer=tf.constant(0), trainable=False)
    #     # self.weight_reg_param_ = tf.Variable(1e-9, name = "reg_param", trainable = False, dtype = tf.float32)
    #     # self.increment_reg_param = tf.cond(self.weight_reg_param_ < 3e-6, true_fn = lambda: tf.assign(self.weight_reg_param_, self.weight_reg_param_ + 1e-10), false_fn = lambda: self.weight_reg_param_)
        
    #     if self.encoder_type_ == 'vgg':
    #         pass
    #         #TODO: to be implemented
    #         # self.height_ , self.width_, self.channel_ = data.shape[1:]
    #         # keys = list(kwargs)
    #         # assert('strides' in keys)
    #         # self.strides_ = kwargs['strides']
    #         # self.kwargs_ = {'strides': self.strides_, 'height': self.height_, 'width': self.width_, 'channel': self.channel_}
            
    #     if self.speaker_type_ == 'agnostic':
    #         self.speaker_ = Agnostic_Speaker(self.encoder_type_, self.num_attributes_, self.dense_len_, self.num_candidates_-1, self.vocabulary_size_, self.temperature_, sess = self.sess_)
    #     else:
    #         self.speaker_ = Informed_Speaker(self.encoder_type_, self.num_attributes_, self.dense_len_, self.num_candidates_-1, self.num_filters_, self.vocabulary_size_, self.temperature_)
        
    #     self.speaker_log_prob_ = self.speaker_.log_prob
    #     self.message_ = self.speaker_.message

    #     self.listener_ = Listener(self.num_attributes_, self.dense_len_, self.num_candidates_, self.message_, self.vocabulary_size_, self.temperature_)
        
    #     # self.speaker_entropy_ = self.speaker_.entropy
    #     self.listener_log_prob_ = self.listener_.log_prob
    #     self.index_match_indicator_ = self.listener_.index_match_indicator

    #     self.reward_ = tf.cast(self.index_match_indicator_, tf.float32) * (self.speaker_log_prob_ + self.listener_log_prob_)
    #     # self.loss_ = tf.reduce_mean(- self.reward_) + 0*self.listener_.reg_loss +  0 * self.speaker_.reg_loss
    #     self.speaker_reg_loss_ = tf.add_n([ tf.nn.l2_loss(v) for v in self.speaker_.reg_varlist_ if 'bias' not in v.name ])
    #     self.listener_reg_loss_ = tf.add_n([ tf.nn.l2_loss(v) for v in self.listener_.reg_varlist_ if 'bias' not in v.name ])
    #     self.weight_loss_ = tf.reduce_mean(- self.reward_)
    #     self.loss_ =  self.weight_loss_ + 1e-6 * self.speaker_reg_loss_ + 1e-6 * self.listener_reg_loss_
    #     self.loss_summary_ = tf.summary.scalar("Loss", self.loss_)
    #     self.optimize_ = tf.train.AdamOptimizer(self.learning_rate_).minimize(self.loss_, global_step=self.global_step_)

    #     self.curr_accr_ = tf.reduce_mean(tf.cast(self.index_match_indicator_, dtype=tf.float64))
    #     tf.get_default_graph().add_to_collection("Accuracy", self.curr_accr_)
    
    #generate batches from a given training dataset

  
    #keeps generating new batches of training data
    def rd_gen_batch(self, is_pretrain = False, gen_from_dataset = False):
        print("---------------------Start random generate data------------------------")
        train_permute = self.train_permutation_
        if is_pretrain:
            train_permute = False
        
        generated_dataset = False
        if gen_from_dataset:
            generated_dataset = True
        # sys.path.insert(1, '../CL_Net/Referential_Game/Geometry3D_4/')
        sys.path.append('../CL_Net/Referential_Game/' + self.concept_folder_)
        from concept import Concept
        game_config = edict({'attributes': range(self.num_attributes_), 'num_distractors': self.num_candidates_, \
            'concept_max_size' : self.max_concept_len_, 'permutation_train' : train_permute, "permutation_test":self.test_permutation_,\
            'generated_dataset' : generated_dataset, 'generated_dataset_path_train' : self.train_dataset_path_, 'generated_dataset_path_test':self.test_dataset_path_, \
            'dataset_attributes_path' : self.dataset_attributes_path_,\
            'feat_dir' : self.pretrain_feature_path_,\
            'img_h' : 128, 'img_w' : 128, 'save_dir' : None} )    
        # game_config = edict({'attributes': range(self.num_attributes_), 'num_distractors': self.num_candidates_, 'concept_max_size' : 5 })    
        concept_space = Concept(game_config,True)
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
                spvs_temp = np.zeros(self.num_attributes_)
               
                spvs_temp[np.array(t_concepts[t_target_idx])] = 1
                t_attr_spvs.append(spvs_temp)
                # print("t_concept_embed: {}".format(t_concept_embed))
                # print("len: {}".format(t_concept_embed.shape))
                # print("t_s_concept_idx_mapping: {}".format(t_s_concept_idx_mapping))
                # print("attr_equiv_class_map: {}".format(attr_equiv_class_mapping))
                # exit()
            #generate teacher's data
            t_target_indices = np.array(t_target_indices)
            t_data = np.array(t_data).reshape(-1, self.input_data_len_)
            t_attr_spvs = np.array(t_attr_spvs)

            # print(t_target_indices + np.arange(0, self.batch_size_ , self.num_candidates_))
            # print(t_data.shape)
            t_targets = t_data[t_target_indices + np.arange(0, self.batch_size_ * self.num_candidates_, self.num_candidates_)] 
            mask = np.ones((self.batch_size_, self.num_candidates_), dtype=bool)
            mask[np.arange(self.batch_size_), t_target_indices] = False
            t_data = t_data.reshape(self.batch_size_, self.num_candidates_, self.input_data_len_)
            t_distractors = t_data[mask].reshape(self.batch_size_, self.num_candidates_-1, self.input_data_len_)
            
            #generate student's data
            s_data = np.array(s_data).reshape(self.batch_size_, self.num_candidates_, self.input_data_len_)
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
            batch['t_distractors'] = t_distractors
            batch['t_target_indices'] = t_target_indices
            batch['s_data'] = s_data
            batch['s_target_indices'] = s_target_indices
            batch['attributes'] = t_attr_spvs
            batch['attr_equiv_class'] = attr_equiv_class_mappings

            yield batch
    
    '''
    to be implemented: self.batch_size_ should be auto adjusted to the nearest factor of self.target_size_
    '''

    def evaluate(self, num_iter, dir_name = None, ckpt_name = None, dataset_attributes_path = None, pretrain_feature_path = None):
        # self._build_model(attr_equiv_class)
        # self.sess_.run(tf.global_variables_initializer())

        #tf.reset_default_graph()
        # print("hello world")
        dir_name = "./" + dir_name+"/"
        test_saver = tf.train.import_meta_graph(dir_name + ckpt_name + '.meta')
        test_saver.restore(self.sess_, dir_name + ckpt_name)
        
        graph = tf.get_default_graph()
        
        speaker_target = graph.get_collection("Speaker_input")[0]
        speaker_distract = graph.get_collection("Speaker_input")[1]
        listener_data = graph.get_collection("Listener_input")[0]
        listener_target_idx = graph.get_collection("Listener_input")[1]
        listener_is_train = graph.get_collection("Listener_input")[2]
        listener_attr_equiv_class = graph.get_collection("Listener_input")[3]
        curr_accr = graph.get_collection("Accuracy")[0]
        # speaker_msg = graph.get_collection('Speaker_input')[2]
        
        #load the test dataset
        sys.path.append('../CL_Net/Referential_Game/' + self.concept_folder_)
        from concept import Concept
        
        # game_config = edict({'attributes': list(range(self.num_attributes_)), 'num_distractors': self.num_candidates_, \
        #     'concept_max_size' : self.max_concept_len_, 'permutation' : test_permute, \
        #     'generated_dataset' : True, 'generated_dataset_path' : test_set_path})    
        
        game_config = edict({'attributes': range(self.num_attributes_), 'num_distractors': self.num_candidates_, \
            'concept_max_size' : self.max_concept_len_, \
            'generated_dataset' : True, 'generated_dataset_path_train' : self.train_dataset_path_, 'generated_dataset_path_test':self.test_dataset_path_, \
            'dataset_attributes_path' : dataset_attributes_path,\
            'feat_dir' : pretrain_feature_path,\
            'img_h' : 128, 'img_w' : 128, 'save_dir' : None, "permutation_train": self.train_permutation_,'permutation_test' : self.test_permutation_} )    

        concept_space = Concept(game_config, False)

        msg_dict = collections.defaultdict(dict)
        test_accr = []
        rtd_accr = []
        hard_accr = []
        rtd_hard_accr = []
        game_mean_coss = []
        for _ in range(num_iter):
            (t_concepts, s_concepts), (t_included_attr, s_included_attr), t_data, s_data, \
            attr_equiv_class_mapping, t_s_concept_idx_mapping, t_target_idx = concept_space.rd_generate_concept()
         
            s_target_idx = np.where(t_s_concept_idx_mapping == t_target_idx)[0]

            t_target = t_data[t_target_idx]
          
            # num_test_data = t_target_idx.shape[0]

            #generate distractors
            t_mask = np.ones((self.num_candidates_), dtype = bool)
            t_mask[ t_target_idx] = False
            t_distractors = t_data[t_mask].reshape(-1, self.num_candidates_-1, self.input_data_len_)
            
            t_cosine = np.sum(t_data * t_target, axis = 1) / (np.sqrt(np.sum(t_data ** 2, axis = 1)) * np.sqrt(np.sum(t_target ** 2)))
            s_cosine = np.sum(s_data * s_data[s_target_idx], axis = 1) / (np.sqrt(np.sum(s_data ** 2, axis = 1)) * np.sqrt(np.sum(s_data[s_target_idx] ** 2)))
            
            # t_cosine_max = t_cosine[np.argsort(t_cosine)[-2]]
            # s_cosine_max = s_cosine[np.argsort(s_cosine)[-2]]
            # game_max_cos = np.mean([t_cosine_max,s_cosine_max])
            game_mean_cosine = (np.mean(t_cosine[np.argsort(t_cosine)[:-1]] ) + np.mean(s_cosine[np.argsort(s_cosine)[:-1]]))/2
            assert(game_mean_cosine <= 1)
            # game_max_coss.append(game_max_cos)
            game_mean_coss.append(game_mean_cosine)

            
            t_target = t_target.reshape(-1,self.input_data_len_)

            fd = {speaker_target: t_target, \
                    speaker_distract : t_distractors,\
                    listener_data: s_data.reshape(-1, self.num_candidates_,self.input_data_len_), \
                    listener_target_idx: s_target_idx, \
                    listener_attr_equiv_class: attr_equiv_class_mapping.reshape(-1, self.num_attributes_), \
                    listener_is_train : False}
            
            accr = self.sess_.run([curr_accr], feed_dict = fd)
           
            # print("t_msg: {}".format(t_msg))
            # print("accr: {}".format(accr))
            # exit()
            # t_msg_idx = np.where(np.array(t_msg) == 1)[0][0]
            # print(t_msg_idx)
            # print("t_target_idx: {}".format(t_target_idx))
            # try :
            #     msg_dict[t_target_idx][t_msg_idx] += 1
            # except KeyError:
            #     msg_dict[t_target_idx][t_msg_idx] = 1


            t_td_dict, _ = concept_space.teaching_dim(t_concepts, t_included_attr)
            s_td_dict, _ = concept_space.teaching_dim(s_concepts, s_included_attr)
            #append rtd=1 accuracy
            if concept_space.recursive_teaching_dim(t_concepts) == 1:
                rtd_accr.append(accr)
                if t_td_dict[t_concepts[t_target_idx]][0] > 1:
                    rtd_hard_accr.append(accr)

            if t_td_dict[t_concepts[t_target_idx]][0] > 1:
                hard_accr.append(accr)
            test_accr.append(accr)

        # for i in range(self.num_candidates_):
        #     print(msg_dict[i])
        print("test accuracy: {}".format(np.mean(test_accr)))
        # print("rtd = 1 accuracy: {}".format(np.mean(rtd_accr)))
        # print("rtd = 1 hard accuracy: {}".format(np.mean(rtd_hard_accr)))
        # print("hard accuracy: {}".format(np.mean(hard_accr)))
        # evaluation_dict = { "game_mean_coss" : game_mean_coss, "test_accr" : test_accr,\
        #     "rtd_accr" : rtd_accr, "rtd_hard_accr" : rtd_hard_accr, "hard_accr" : hard_accr}
        # np.save(dir_name+"cos_dict.npy", evaluation_dict)

        # game_mean_coss = np.array(game_mean_coss)
        test_accr = np.array(test_accr)
        top_ten_idx = np.argsort(game_mean_coss)[-int(len(game_mean_coss)/10):]
        print("top 10% test accuracy: {}".format(np.mean(test_accr[top_ten_idx])))
        # print("top 10% /rtd=1 test accuracy: {}".format(np.mean(rtd_accr[top_ten_idx])))
        # print("top 10% /rtd=1 hard test accuracy: {}".format(np.mean(rtd_hard_accr[top_ten_idx])))
        # print("top 10% /hard test accuracy: {}".format(np.mean(hard_accr[top_ten_idx])))



        # test_accr = np.array(test_accr)
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
        
    def train(self,  max_iter, folder_name,  pretrain = True, dataset_attributes_path = None, pretrain_feature_path = None):
        #create the graph
        self._build_model()
        
        self.pretrain_ = pretrain

        pre_speaker_saver = tf.train.Saver()
        pre_listener_saver = tf.train.Saver()
        train_saver = tf.train.Saver(max_to_keep = 100)

        loader = tf.train.Saver()
        train_summary_writer = tf.summary.FileWriter("./summary", self.sess_.graph)

        # checkpoint = tf.train.get_checkpoint_state(os.path.dirname('game'))
        # if checkpoint and checkpoint.model_checkpoint_path:
        #     saver.restore(self.sess_, checkpoxint.model_checkpoint_path)
        # total_iter = self.global_step_.eval()

        self.sess_.run(tf.global_variables_initializer())
        total_accr = tf.constant(0, dtype=tf.float64)        
        current_iter = 0
                  
        self.dataset_attributes_path_ = dataset_attributes_path
        self.pretrain_feature_path_ = pretrain_feature_path
        
        self.batch_generator_ = self.rd_gen_batch


        # if self.train_permutation_:
        #         if self.pretrain_:
        #             train_folder_name = "./checkpoints_train_" + dataset_name + "_with_permute_with_pretrain/"
        #         else:
        #             train_folder_name = "./checkpoints_train_" + dataset_name + "_with_permute_no_pretrain/"

        # else:
        #     if self.pretrain_:
        #         train_folder_name = "./checkpoints_train_" + dataset_name + "_without_permute_with_pretrain/"
        #     else:
        #         train_folder_name = "./checkpoints_train_" + dataset_name + "_without_permute_no_pretrain/"

        train_folder_name = "./" + folder_name +"/"
                    
        #create folders to save pretrain and training checkpoints if not exists
        if self.pretrain_ and not os.path.exists(train_folder_name):
            pretrain_speaker_folder_name = "./pretrain_" + folder_name + "_speaker/"
            pretrain_listener_folder_name = "./pretrain_" + folder_name + "_listener/"

            # if self.train_permutation_:
            #     pretrain_listener_folder_name = "./pretrain_" + folder_name + "_listener_with_permute/"
            #     pretrain_speaker_folder_name = "./checkpoints_pretrain_" + dataset_name + "_speaker_with_permute/"
            # else:
            #     pretrain_listener_folder_name = "./checkpoints_pretrain_" + dataset_name + "_listener_without_permute/"
            #     pretrain_speaker_folder_name = "./checkpoints_pretrain_" + dataset_name + "_speaker_without_permute/"
            
            if not os.path.exists(pretrain_speaker_folder_name):
                os.makedirs(pretrain_speaker_folder_name)
            if not os.path.exists(pretrain_listener_folder_name):
                os.makedirs(pretrain_listener_folder_name)


            #pretrain speaker
            pretrain_speaker_file_name = pretrain_speaker_folder_name + self.speaker_type_ + '-' + str(max_iter)
            # pretrain_speaker_loader = tf.train.get_checkpoint_state(pretrain_folder)
            try:
                loader.restore(self.sess_, pretrain_speaker_file_name)
            except ValueError: 
                print("Fail to load checkpoint {}".format(pretrain_speaker_file_name))
                print("Start pretraining speaker")
                for batch in self.batch_generator_(is_pretrain = True):
                    fd = {self.speaker_.target_ : batch['t_targets'],\
                        self.speaker_.ori_distract_ : batch['t_distractors'],\
                        self.pred_attr_spvs : batch['attributes']}
                    
                    _, pretrain_speaker_loss, pred_attr = self.sess_.run([self.speaker_pretrain_op_, self.pretrain_speaker_loss_, self.pred_attr_], feed_dict = fd)
                    if current_iter % 500 == 0:
                        print("iter {} pretrain speaker loss: {}".format(current_iter, pretrain_speaker_loss))
                        # print("pred_attr: {}".format(pred_attr))
                        # print("attr_spvs: {}".format(batch['attributes']))
                    if current_iter % 5000 == 0:
                        pre_speaker_saver.save(self.sess_, pretrain_speaker_folder_name + self.speaker_type_ + '-'+str(current_iter))
                        
                    if current_iter == max_iter:
                        break
                    current_iter += 1 
            else:
                print("Successfully loaded checkpoint: {}".format(pretrain_speaker_file_name))

        
            #pretrain listener
            current_iter = 0
            pretrain_listener_filename = pretrain_listener_folder_name + self.speaker_type_ + '-listener-' + str(max_iter)
            # pretrain_listener_total_accr = tf.constant(0, dtype=tf.float64)        
            accuracies = []
            try:
                loader.restore(self.sess_, pretrain_listener_filename)
            except ValueError: 
                print("Fail to load checkpoint {}".format(pretrain_listener_filename))
                print("Start pretraining listener")

                for batch in self.batch_generator_(is_pretrain = True):
                    fd = {
                        self.speaker_.target_: batch['t_targets'], \
                        self.speaker_.ori_distract_ : batch['t_distractors'], \
                        self.listener_.ori_data_ : batch['s_data'],\
                        self.listener_.target_idx_ : batch['s_target_indices'],\
                        self.listener_.attr_equiv_class_ : batch['attr_equiv_class'],\
                        self.listener_.is_train_ : True}

                    # print("t_target_indices: {}".format(batch["t_target_indices"][0]))
                    # print("s_target_indices: {}".format(batch["s_target_indices"][0]))
                    # print("attr_equiv_class: {}".format(batch["attr_equiv_class"][0]))
                    # weights = self.sess_.run([self.kernel_1], feed_dict=fd)
                    # print("speaker weights before pretrain listener: {}".format(weights))
                    _, pretrain_listener_loss, pre_lis_cur_accr, sp_logits, sp_msgs, ls_msgs = self.sess_.run([self.listener_pretrain_op_, self.pretrain_listener_loss_, self.pretrain_listener_accr_, self.speaker_logits_, self.speaker_msg_,self.listener_msg_ ], feed_dict = fd)
                    accuracies.append(pre_lis_cur_accr)
                    # pretrain_listener_total_accr += pre_lis_cur_accr
                    # print("weights after pretrain: {}".format(feature_weights))
                    # print("targets: {}".format(batch['targets']))
                    # print("speaker logits: {}".format(sp_logits))
                    # print("speaker probs: {}".format(sp_probs))
                    # print("speaker msgs: {}".format(sp_msgs))
                    # print("listener message: {}".format(ls_msgs))
                    # exit()
                    if current_iter % 500 == 0:
                        print("iter {} pretrain listener loss: {}, cur_acc: {}".format(current_iter, pretrain_listener_loss, pre_lis_cur_accr))
                        print("mean acc: {}".format(np.mean(accuracies)))
                        accuracies = []
                    if current_iter % 5000 == 0:
                        pre_listener_saver.save(self.sess_, pretrain_listener_folder_name + self.speaker_type_ + '-listener-' + str(current_iter))

                    if current_iter == max_iter:
                        break
                    current_iter += 1

                    # if current_iter == 1:
                    #     exit()
            else:
                print("Successfully loaded checkpoint: {}".format(pretrain_listener_filename))
       

    
        if not os.path.exists(train_folder_name):
            os.makedirs(train_folder_name)

        #normal training
        accuracies = []
        current_iter = resume_iter = 0

        latest_ckpt_name = tf.train.latest_checkpoint(train_folder_name)
        if latest_ckpt_name is not None:
            try:
                loader.restore(self.sess_, latest_ckpt_name)
            except ValueError: 
                print("Fail to load checkpoint {}".format(latest_ckpt_name))
                
            else:
                print("Successfully loaded checkpoint: {}".format(latest_ckpt_name))
                resume_iter = int(latest_ckpt_name.split('-')[1])
        else:
            print("No training checkpoints to be loaded, training from start")
        
        
      
        for batch in self.batch_generator_(gen_from_dataset = True):
            fd = {self.speaker_.target_: batch['t_targets'], \
                  self.speaker_.ori_distract_ : batch['t_distractors'],\
                  self.listener_.ori_data_: batch['s_data'], \
                  self.listener_.target_idx_: batch['s_target_indices'],\
                  self.listener_.attr_equiv_class_ : batch['attr_equiv_class'],\
                  self.listener_.is_train_ : True}
            spk_msg = self.sess_.run(self.speaker_msg_, feed_dict = fd)
         
            _,  loss,summary, curr_accr, cur_reward = self.sess_.run([self.train_op_, self.loss_, self.loss_summary_, self.curr_accr_, self.reward_], feed_dict=fd)
            accuracies.append(curr_accr)
            # rewards.append(cur_reward)
            # total_accr = total_accr + curr_accr
            # print("t_target: {}".format(batch['t_targets'][2]))
            # print("s_data: {}".format(batch['s_data'][2]))
            # print("s_target_indices: {}".format(batch['s_target_indices'][2]))
            # exit()
            # train_summary_writer.add_summary(summary, current_iter)
            # self.sess_.run(self.speaker_.d2m_.get_clean_var_operators())
            if current_iter % 500 == 0:
                print("iter: {}, loss: {}".format(current_iter, loss))
                # print("weight_loss: {}, speaker_reg_loss: {}, listener_reg_loss: {}".format(weight_loss, speaker_loss, listener_loss))
                # print("regularization param: {}".format(reg_par));
                print("current accuracy: {}".format(curr_accr))
                print("mean accuracy: {}".format(np.mean(accuracies)))
                # print("current reward: {}".format(cur_reward))
                # print("mean accuracy: {}".format(np.mean(accuracies)))
                # print("mean reward: {}".format(np.mean(rewards)))
                accuracies = []
                # rewards = []
                # print(lis_is_train)
                if current_iter % 2500 == 0:
                    train_saver.save(self.sess_, train_folder_name + "checkpoint", global_step = self.global_step_)
                if current_iter == 200000:
                    break
            resume_iter += 1
            current_iter += 1
            
