import os
import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict

import pdb

class Agent:
    def __init__(self, sess, train_config, game_config, visual_config, belief_config, value_config, role):
        self.sess_ = sess
        self.num_distractors_ = game_config.num_distractors
        self.message_space_size_ = game_config.message_space_size
        self.visual_config_ = visual_config
        self.belief_config_ = belief_config
        self.game_config_ = game_config
        self.train_config_ = train_config
        self.value_config_ = value_config
        self.role_ = role

        self.max_entropy = None

        self.global_step_ = tf.Variable(0, trainable = False)
        ################
        # Placeholders #
        ################
        with tf.variable_scope(self.role_):
            if self.visual_config_.type == 'Feat':
                self.distractors_ = tf.placeholder(tf.float32, name = 'distractors', 
                                                     shape = [None, self.num_distractors_, self.visual_config_.feat_dim])
                self.distractors_tensor_ = tf.expand_dims(self.distractors_, 2)
            else:
                self.distractors_ = tf.placeholder(tf.float32, name = 'distractors', 
                                                     shape = [None, self.num_distractors_,
                                                              self.visual_config_.img_h,
                                                              self.visual_config_.img_w, 3])
            self.message_ = tf.placeholder(tf.float32, shape = [None, self.message_space_size_], name = 'message')

            self.teacher_belief_ = tf.placeholder(tf.float32, shape = [None, self.num_distractors_], name = 'teacher_belief')
            self.student_belief_ = tf.placeholder(tf.float32, shape = [None, self.num_distractors_], name = 'student_belief')
            self.belief_spvs_ = tf.placeholder(tf.float32, shape = [None, self.num_distractors_], name = 'belief_spvs')
            self.value_spvs_ = tf.placeholder(tf.float32, shape = [None], name = 'value_spvs')
        #######################
        #  Perception Module  #
        #######################
        with tf.variable_scope('%s_Feature_Extract' % self.role_):
            if self.visual_config_.type != 'Feat':
                exec('from Modules.perception import %sEncoder as VE' % self.visual_config_.type, globals())
                self.perception_core_ = VE(self.distractors_, self.visual_config_)
            else:
                self.perception_core_ = edict({'visual_features_': tf.identity(self.distractors_tensor_)})
        ########################
        # Belief Update Module #
        ########################
        with tf.variable_scope('%s_Belief_Update' % self.role_):
            exec('from Modules.meditation import %sBU as BU' % self.belief_config_.type, globals())
            self.meditation_core_ = BU(self.student_belief_,
                                       self.perception_core_.visual_features_,
                                       self.message_, self.belief_config_)       
        #########################
        # Belief Supervise Loss #
        #########################
        self.reg_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)\
                             if v.name.startswith('%s_Belief' % self.role_) or v.name.startswith('%s_Feature' % self.role_)]
        
        self.regularization_ = self.train_config_.regularization_param * tf.add_n([ tf.nn.l2_loss(v) for v in self.reg_varlist_ if 'bias' not in v.name ])

        self.cross_entropy_1_ = -1 * tf.reduce_mean(tf.reduce_sum(tf.multiply(self.belief_spvs_, tf.math.log(self.meditation_core_.posterior_ + 1e-9)), axis = 1))
        self.cross_entropy_2_ = -1 * tf.reduce_mean(tf.reduce_sum(tf.multiply(self.meditation_core_.posterior_, tf.math.log(self.belief_spvs_ + 1e-9)), axis = 1))
        self.cross_entropy_ = self.cross_entropy_1_ + self.cross_entropy_2_ + self.regularization_
        ########################
        # Value Network Module #
        ########################
        from Modules.value import ValueNetwork as VN
        with tf.variable_scope('%s_Value_Network' % self.role_):
            self.value_core_ = VN(self.teacher_belief_, self.meditation_core_.posterior_, self.perception_core_.visual_features_, self.value_config_)
        
        self.value_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('%s_Value' % self.role_)]
        self.regularization_value_ = 1e-5 * tf.add_n([ tf.nn.l2_loss(v) for v in self.value_varlist_ if 'bias' not in v.name ])
       
        self.value_net_loss_pre_ = tf.square(self.value_core_.value_ - self.value_spvs_)
        self.success_mask_ = tf.to_float(tf.math.greater(self.value_spvs_, 0.0))
        self.fail_mask_ = tf.to_float(tf.math.greater(0.0, self.value_spvs_))
        self.imbalance_penalty_ = self.success_mask_ + self.fail_mask_ * tf.div_no_nan(tf.reduce_sum(self.success_mask_), tf.reduce_sum(self.fail_mask_))
        # self.value_net_loss_ = tf.reduce_mean(self.value_net_loss_pre_ * tf.to_float(self.value_net_loss_pre_ > 0.05) * self.imbalance_penalty_) + self.regularization_q_
        self.value_net_loss_ = tf.reduce_mean(self.value_net_loss_pre_ * self.imbalance_penalty_) + self.regularization_value_
        ################
        ## DL devices ##
        ################
        self.belief_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('%s_Belief' % self.role_)]
        self.feature_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('%s_Feature' % self.role_)]
        self.value_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('%s_Value' % self.role_)]
        
        if self.train_config_.feature_decay is not None:
            self.feature_learning_rate_ = tf.train.exponential_decay(self.train_config_.feature_lr,
                                                                     self.global_step_,
                                                                     self.train_config_.feature_decay,
                                                                     0.1, staircase = True)
        else:
            self.feature_learning_rate_ = tf.constant(self.train_config_.feature_lr)
        if self.train_config_.belief_decay is not None:
            self.belief_learning_rate_ = tf.train.exponential_decay(self.train_config_.belief_lr,
                                                                    self.global_step_,
                                                                    self.train_config_.belief_decay,
                                                                    0.1, staircase = True)
        else:
            self.belief_learning_rate_ = tf.constant(self.train_config_.belief_lr)
        self.value_learning_rate_ = 1e-4#tf.train.exponential_decay(1e-4, self.global_step_, 200000, 0.1, staircase = True)
        
        self.feature_update_opt_ = tf.train.AdamOptimizer(learning_rate = self.feature_learning_rate_)
        self.belief_update_opt_ = tf.train.AdamOptimizer(learning_rate = self.belief_learning_rate_)
        self.value_update_opt_ = tf.train.AdamOptimizer(learning_rate = self.value_learning_rate_)

        self.belief_train_op_ = self.belief_update_opt_.minimize(self.cross_entropy_, 
                                                                 var_list = self.belief_varlist_,
                                                                 global_step = self.global_step_)
        self.value_train_op_ = self.value_update_opt_.minimize(self.value_net_loss_,
                                                               var_list = self.value_varlist_,
                                                               global_step = self.global_step_)

        if self.visual_config_.type == 'Feat':
            self.bf_op_ = self.belief_train_op_
        else:
            self.feature_train_op_ = self.feature_update_opt_.minimize(self.cross_entropy_,
                                                                       var_list = self.feature_varlist_,
                                                                       global_step = self.global_step_)
            self.bf_op_ = tf.group(self.belief_train_op_, self.feature_train_op_)
        
        if belief_config.batch_norm:
            self.update_op_ = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if role in op.name]
            self.bf_op_ = tf.group([self.bf_op_, self.update_op_])
        
        if belief_config.batch_norm:
            self.batch_norm_varlist_ = [v for v in tf.global_variables() if v.name.startswith('%s_Belief_Update/BatchNorm' % role) and 'Adam' not in v.name]
            self.batch_mean = [var for var in tf.global_variables() if var.op.name == '%s_Belief_Update/BatchNorm/moving_mean' % role][0]
            self.batch_variance = [var for var in tf.global_variables() if var.op.name == '%s_Belief_Update/BatchNorm/moving_variance' % role][0]
        else:
            self.batch_norm_varlist_ = []

        self.pretrain_saver_ = tf.train.Saver(list(set(self.belief_varlist_ + self.feature_varlist_ + self.batch_norm_varlist_)))

    def train_belief_update(self, data_batch, quick = False):
        fd = {self.student_belief_: data_batch['prev_belief'],
              self.message_: data_batch['message'],
              self.distractors_: data_batch['distractors'],
              self.belief_spvs_: data_batch['new_belief']}
        if quick:
            _, cross_entropy, belief_pred = self.sess_.run([self.bf_op_, 
                                        self.cross_entropy_,
                                        self.meditation_core_.posterior_], feed_dict = fd)
            return cross_entropy, belief_pred
        
        else:
            if self.belief_config_.batch_norm:
                run_batch_norm = lambda: self.sess_.run([self.meditation_core_.likelihood_, 
                                            self.meditation_core_.visual_feat_, 
                                            self.bf_op_, 
                                            self.cross_entropy_,
                                            self.meditation_core_.posterior_,
                                            self.batch_mean,
                                            self.batch_variance,
                                            self.feature_learning_rate_,
                                            self.belief_learning_rate_], feed_dict = fd)
                debug_msg1, debug_msg2, _, cross_entropy, belief_pred, batch_mean, batch_var, flr, blr = run_batch_norm()
            else:
                run_no_batch_norm = lambda: self.sess_.run([self.meditation_core_.likelihood_, 
                                            self.meditation_core_.visual_feat_, 
                                            self.bf_op_, 
                                            self.cross_entropy_,
                                            self.meditation_core_.posterior_,
                                            self.feature_learning_rate_,
                                            self.belief_learning_rate_], feed_dict = fd)
                debug_msg1, debug_msg2, _, cross_entropy, belief_pred, flr, blr = run_no_batch_norm()
                batch_mean = batch_var = None
            # print(debug_msg2[0])
            return cross_entropy, belief_pred, flr, blr, batch_mean, batch_var, [debug_msg1, debug_msg2]

    def pretrain_bayesian_belief_update(self, concept_generator, silent = False):
        
        pretrain_ckpt_dir = 'Bayesian_Pretrain_%s' % self.role_
        pretrain_ckpt_dir = os.path.join(self.train_config_.save_dir, pretrain_ckpt_dir)
        if not os.path.exists(pretrain_ckpt_dir):
            os.makedirs(pretrain_ckpt_dir)

        ckpt = tf.train.get_checkpoint_state(pretrain_ckpt_dir)

        if ckpt:
            try:
                self.pretrain_saver_.restore(self.sess_, ckpt.model_checkpoint_path)
                print('Loaded %s (feature extract and) belief update ckpt from %s' % (self.role_, pretrain_ckpt_dir))
            except:
                self.pretrain_belief_saver_ = tf.train.Saver(list(set(self.belief_varlist_ + self.batch_norm_varlist_)))
                self.pretrain_belief_saver_.restore(self.sess_, ckpt.model_checkpoint_path)
            print('Loaded %s belief update ckpt from %s' % (self.role_, pretrain_ckpt_dir))
            train_steps = self.train_config_.continue_steps if self.train_config_.get('continue_steps') else 0
            pretrain_batch_size = -1 if train_steps == 0\
                                     else self.train_config_.pretrain_batch_size
        else:
            print('Cannot loaded %s belief update ckpt from %s' % (self.role_, pretrain_ckpt_dir))
            pretrain_batch_size = self.train_config_.pretrain_batch_size
            train_steps = self.train_config_.pretrain_steps

        accuracies = []
        l1_diffs = []
        bayesian_wrongs = []
        for ts in range(int(train_steps)):
            data_batch = concept_generator.generate_batch(pretrain_batch_size, self.role_)
            if ts % self.train_config_.print_period == 0 and not silent:
                cross_entropy, belief_pred, flr, blr, batch_mean, batch_var, debug_msg = self.train_belief_update(data_batch)
            else:
                cross_entropy, belief_pred  = self.train_belief_update(data_batch, quick = True)
            
            l1_diff = np.sum(abs(belief_pred - data_batch['new_belief']), axis = 1)
            correct = (l1_diff <= 5e-2)
            bayesian_wrong = np.mean(np.sum((data_batch['new_belief'] == 0) * (belief_pred > 1e-5), axis = 1) > 0)
            accuracies.append(np.mean(correct))
            l1_diffs.append(np.mean(l1_diff))
            bayesian_wrongs.append(bayesian_wrong)
            if np.sum(np.isnan(belief_pred)) != 0:
                print(belief_pred)
                pdb.set_trace()

            if ts % self.train_config_.print_period == 0 and not silent:
                print('[%s%d] batch mean cross entropy: %f, mean accuracies: %f, mean l1: %f, bayesian wrong: %f'\
                    % (self.role_[0], ts + 1, cross_entropy, np.mean(accuracies), np.mean(l1_diffs), np.mean(bayesian_wrongs)))
                belief_var_1d = self.sess_.run(self.meditation_core_.belief_var_1d_)
                if self.belief_config_.type == 'Explicit' or self.belief_config_.type == 'OrderFree':
                    boltzman_beta = self.sess_.run(self.meditation_core_.boltzman_beta_)
                    print('boltzman_beta: %f, belief_var_1d: %f' % (boltzman_beta, belief_var_1d))
                else:
                    print('belief_var_1d: %f' % belief_var_1d)
                if self.belief_config_.batch_norm:
                    print('mean: %f, var: %f' % (np.mean(batch_mean), np.mean(batch_var)))

                if self.belief_config_.type == 'OrderFree':
                    entropy = np.mean(-1 * np.sum(belief_pred * np.log(belief_pred + 1e-10), axis = 1))
                    print('entropy of belief predicted: %f' % entropy, end = ', ')
                    if self.max_entropy is None:
                        self.max_entropy = float(-1 * self.num_distractors_ * (1 / self.num_distractors_) * np.log(1 / self.num_distractors_))
                    print('max entropy: %f' % self.max_entropy)

                if np.mean(accuracies) >= 0.0:
                    #idx = np.random.randint(pretrain_batch_size)
                    idx = 3
                    print('learning rate: %f, %f' % (flr, blr))
                    for i in range(idx):
                        print('\t target:', data_batch['new_belief'][i, :])
                        print('\t predict', belief_pred[i, :])
                        # print('\t likelihood', debug_msg[0][i, ...])
                        # print('\t before softmax', debug_msg[1][i, ...])
                accuracies = []
                l1_diffs = []
                bayesian_wrongs = []
            if (ts + 1) % self.train_config_.save_period == 0 or ts == train_steps - 1:
                self.total_saver_.save(self.sess_, os.path.join(pretrain_ckpt_dir,
                                       '%s_%s_%d' % (self.game_config_.dataset,
                                                     self.visual_config_.type,
                                                     self.game_config_.num_distractors)),
                                       global_step = ts + 1)
                print('Saved %s belief update ckpt to %s after %d training' % (self.role_, pretrain_ckpt_dir, ts))