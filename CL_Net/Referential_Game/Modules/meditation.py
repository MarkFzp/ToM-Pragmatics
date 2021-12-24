import tensorflow as tf

class BeliefUpdate:
    def __init__(self, prev_belief, input_tensor, msg_tensor, config_dict):
        self.fc_layer_info_ = config_dict.fc_configs
        self.batch_norm = config_dict.batch_norm
        self.num_distractors_ = prev_belief.shape[1]
        self.prior_ = prev_belief #[None, num_distractors]
        self.inputs_ = input_tensor # [None, num_distractors, 1, feat_dim]
        self.msg_tensor_ = msg_tensor #one-hot msg: [None, msg_space_size]
        self.belief_var_1d_ = tf.exp(tf.Variable(initial_value = config_dict.belief_var_1d, trainable = True,
                                                    dtype = tf.float32, name = 'invalid_bias'))
        self.initializer_ = tf.random_normal_initializer(mean = 0, stddev = 1e-2)#tf.contrib.layers.variance_scaling_initializer()

        if self.batch_norm:
            self.feat_tensors_ = [tf.contrib.layers.batch_norm(self.inputs_, is_training = False)]
        else:
            self.feat_tensors_ = [self.inputs_]

        for idx, out_dim in enumerate(self.fc_layer_info_):
            # fc = tf.layers.conv2d(self.feat_tensors_[-1], out_dim, kernel_size=1, strides=1, padding='VALID', 
            #                       activation = tf.nn.leaky_relu, kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 5e-2))
            fc = tf.layers.conv2d(self.feat_tensors_[-1], out_dim, kernel_size=1, strides=1, padding='VALID', 
                        activation = tf.nn.leaky_relu, kernel_initializer = self.initializer_)
            self.feat_tensors_.append(fc)
    
    def build_context_conv(self):
        assert(self.num_dims_[-1] == self.num_distractors_)
        for idx, num_filter in enumerate(self.num_filters_):
            tensor_rows = []
            kernel_dim = self.num_distractors_ if idx == 0 else self.num_dims_[idx - 1]
            non_linear_func = tf.nn.leaky_relu if idx != len(self.num_filters_) - 1 else None
            for _ in range(self.num_dims_[idx]):
                # tensor_rows.append(tf.layers.conv2d(self.feat_tensors_[-1], num_filter, kernel_size = [kernel_dim, 1],
                #                                     kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-1),
                #                                     padding = 'valid', activation = non_linear_func))
                tensor_rows.append(tf.layers.conv2d(self.feat_tensors_[-1], num_filter, kernel_size = [kernel_dim, 1],
                                                    kernel_initializer = self.initializer_,
                                                    padding = 'valid', activation = non_linear_func))

            feat_tensor = tf.concat(tensor_rows, axis = 1)
            self.feat_tensors_.append(feat_tensor)
        
        self.visual_feat_ = tf.squeeze(self.feat_tensors_[-1], axis = 2, name = "2d_visual_feature")
        self.msg_indices_ = tf.where(tf.not_equal(self.msg_tensor_, 0))
        self.likelihood_ = None

    def build(self):
        self.posterior_raw_ = tf.multiply(self.prior_, self.likelihood_)

        self.posterior_full_ = tf.concat([self.posterior_raw_, self.belief_var_1d_ * tf.slice(tf.ones_like(self.posterior_raw_), [0, 0], [-1, 1])], axis = 1)	
        self.posterior_full_norm_ = tf.div_no_nan(self.posterior_full_, tf.reduce_sum(self.posterior_full_, axis = 1, keepdims = True))
        self.posterior_ = tf.slice(self.posterior_full_norm_, [0, 0], [-1, self.num_distractors_])

class ExplicitBU(BeliefUpdate):
    def __init__(self, prev_belief, input_tensor, msg_tensor, config_dict):
        BeliefUpdate.__init__(self, prev_belief, input_tensor, msg_tensor, config_dict)
        self.num_filters_ = config_dict.num_filters
        self.num_dims_ = config_dict.num_dims
        self.build_context_conv()

        self.boltzman_beta_ = tf.Variable(initial_value = config_dict.boltzman_beta, trainable = False,
                                          dtype = tf.float32, name = 'boltzman_beta')
        
        self.compatibility_ = tf.nn.softmax(self.boltzman_beta_ * self.visual_feat_)			
        
        self.likelihood_ = tf.gather_nd(tf.transpose(self.compatibility_, perm = [0, 2, 1]), self.msg_indices_)
        self.build()

class ImplicitBU(BeliefUpdate):
    def __init__(self, prev_belief, input_tensor, msg_tensor, config_dict):
        BeliefUpdate.__init__(self, prev_belief, input_tensor, msg_tensor, config_dict)
        self.num_filters_ = config_dict.num_filters
        self.num_dims_ = config_dict.num_dims
        self.build_context_conv()
        # self.word_embedding_ = tf.get_variable(shape = [config_dict.num_filters[-1],
        #                                        config_dict.message_space_size], name = 'word_embedding',
        #                                        initializer = tf.initializers.random_normal(mean = 0, stddev = 1e-2))
        self.word_embedding_ = tf.get_variable(shape = [config_dict.num_filters[-1],
                                               config_dict.message_space_size], name = 'word_embedding',
                                               initializer = self.initializer_)
        
        self.msg_embeddings_ = tf.expand_dims(tf.transpose(tf.gather(self.word_embedding_, self.msg_indices_[:, 1], axis = 1)), 1)
        self.compatibility_ = tf.reduce_sum(tf.multiply(self.msg_embeddings_, self.visual_feat_), axis = 2)
        self.likelihood_ = tf.nn.sigmoid(self.compatibility_)
        self.build()

class OrderFreeBU(BeliefUpdate):
    def __init__(self, prev_belief, input_tensor, msg_tensor, config_dict):
        BeliefUpdate.__init__(self, prev_belief, input_tensor, msg_tensor, config_dict)
        self.context_length_ = config_dict.context_length
        self.single_length_ = config_dict.single_length
        for j, cl in enumerate(self.context_length_):
            context = tf.tile(tf.reduce_sum(self.feat_tensors_[-1], axis = 1, keepdims = True), [1, self.num_distractors_, 1, 1])
            context_fc = tf.layers.conv2d(context, cl, kernel_size = 1, strides = 1, padding='VALID', 
                        activation = tf.nn.leaky_relu, kernel_initializer = self.initializer_)
            with_context = tf.concat([self.feat_tensors_[-1], context_fc], axis = 3)
            self.feat_tensors_.append(tf.layers.conv2d(with_context, self.single_length_[j],
                                                       kernel_size = 1, strides = 1, padding='VALID', 
                                                       activation = tf.nn.leaky_relu if j != len(self.context_length_) - 1 else None,
                                                       kernel_initializer = self.initializer_))

        # self.feat_tensors_.append(tf.layers.conv2d(self.feat_tensors_[-1], config_dict.message_space_size, 
        #                                                kernel_size = 1, strides = 1, padding='VALID', 
        #                                                activation = None,
        #                                                kernel_initializer = self.initializer_))
        
        self.visual_feat_ = tf.squeeze(self.feat_tensors_[-1], axis = 2, name = "2d_visual_feature")
        self.msg_indices_ = tf.where(tf.not_equal(self.msg_tensor_, 0))

        self.boltzman_beta_ = tf.Variable(initial_value = config_dict.boltzman_beta, trainable = False,
                                          dtype = tf.float32, name = 'boltzman_beta')
        
        self.compatibility_ = tf.nn.softmax(self.boltzman_beta_ * self.visual_feat_)
        
        self.likelihood_ = tf.gather_nd(tf.transpose(self.compatibility_, perm = [0, 2, 1]), self.msg_indices_)
        self.build()


if __name__ == "__main__":
    main()