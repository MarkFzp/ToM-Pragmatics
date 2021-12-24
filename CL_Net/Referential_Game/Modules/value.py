import tensorflow as tf

class ValueNetwork:
    def __init__(self, gt_belief, query_belief, input_tensor, config_dict):
        self.num_distractors_ = gt_belief.shape[1]
        self.gt_belief_ = gt_belief #[None, num_distractors]
        self.inputs_ = input_tensor # [None, num_distractors, 1, feat_dim]
        self.query_belief_ = query_belief #one-hot msg: [None, num_distractors]
        self.null_belief_penalty_ = tf.Variable(initial_value = config_dict.null_belief_penalty, trainable = True,
                                                dtype = tf.float32, name = 'nul_belief_penalty')
        
        self.fc_layer_info_ = config_dict.fc_layer_info_
        self.cnn_layer_info_ = config_dict.cnn_layer_info_
        assert(self.cnn_layer_info_[-1][0] == 1)

        self.feat_tensors_ = [input_tensor]
        for out_dim in self.fc_layer_info_:
            fc = tf.layers.conv2d(self.feat_tensors_[-1], out_dim, kernel_size=1, strides=1, padding='VALID', 
                                  activation=tf.nn.leaky_relu, kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2))
            self.feat_tensors_.append(fc)
        self.feat_tensors_.append(tf.multiply(self.feat_tensors_[-1], tf.expand_dims(tf.expand_dims(self.query_belief_, -1), -1)))
        
        for idx, (num_filter, num_dim) in enumerate(self.cnn_layer_info_):
            tensor_rows = []
            kernel_dim = self.num_distractors_ if idx == 0 else self.cnn_layer_info_[idx - 1][1]
            non_linear_func = tf.nn.leaky_relu if idx != len(self.cnn_layer_info_) - 1 else None
            for _ in range(num_dim):
                tensor_rows.append(tf.layers.conv2d(self.feat_tensors_[-1], num_filter, kernel_size = [kernel_dim, 1],
                                                    kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1e-2),
                                                    padding = 'valid', activation = non_linear_func))

            feat_tensor = tf.concat(tensor_rows, axis = 1)
            self.feat_tensors_.append(feat_tensor)
        
        self.value_ = tf.reduce_sum(tf.multiply(tf.squeeze(self.feat_tensors_[-1]), self.gt_belief_), axis = 1) +\
					  (1 - tf.reduce_sum(self.query_belief_, axis = 1)) * self.null_belief_penalty_
