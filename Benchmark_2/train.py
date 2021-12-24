import tensorflow as tf
import numpy as np
from Game import Game
import sys
import os
# sys.path.insert(1, '../CL_Net/Referential_Game/Number_Set')
# from concept import Concept
from easydict import EasyDict as edict

USE_GPU = False

configuration = None
if not USE_GPU:
    # [for CPU training]
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    configuration = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    configuration.gpu_options.allow_growth = True
tf.reset_default_graph()
# tf.set_random_seed(1024)
# np.random.seed(1024)
#np.set_printoptions(threshold=np.nan)

def main():
    config_file = sys.argv[1]
    config = {}

    with open(config_file) as f:
        for line in f:
            setting = line.split('=')
            if setting[0] == "train_permute" or setting[0] == "pretrain" or setting[0] == "test_permute":
                config[setting[0]] = setting[1].strip('\n') == "True"
            elif setting[0] == "max_concept_len" or setting[0] == "num_attributes" or setting[0] == "input_data_len" or setting[0] == "num_candidates":
                config[setting[0]] = int(setting[1].strip('\n'))
            else:
                config[setting[0]] = setting[1].strip('\n')

    max_concept_len = config["max_concept_len"]
    #the lengh of data
    input_data_len = config["input_data_len"]
    #the number of attributes a data has 
    vocabulary_size =  num_attributes = config["num_attributes"]
   
    #number of candicates a training data has
    num_candidates = config["num_candidates"]

    dense_len = 50
    num_filter = 20
    # message_len = 4
    batch_size = 100
    num_epoches = 1
    # iteration = 1000
    learning_rate = 1e-3
    
    temperature = 10
   

    with tf.Session(config=configuration) as sess:
    # sess = tf.Session(config=config)
        
        #game = Game(sess, 'fc', 'informed', batch_size, dense_len, data, target_indices, num_epoches, vocabulary_size, num_filters = num_filter, learning_rate=learning_rate, temperature=temperature, is_train = True)
        game = Game(sess, 'fc', 'informed',input_data_len, batch_size, max_concept_len, num_attributes, num_candidates, dense_len, num_epoches ,\
            vocabulary_size = vocabulary_size, num_filters = num_filter, learning_rate=learning_rate, temperature=temperature,\
            concept_folder = config["concept_folder"], train_permutation= config["train_permute"], test_permutation=config["test_permute"],\
                train_set_path = config["train_dataset_path"], test_set_path = config["test_dataset_path"])

        game.train(100000, config["test_folder_path"], pretrain = config["pretrain"], \
            dataset_attributes_path = config["dataset_attributes_path"],\
            pretrain_feature_path = config["pretrain_feature_path"])
        
     
        #for all test set
       
        game.evaluate(100000, \
            dir_name = config['test_folder_path'],\
                ckpt_name=config["ckpt_name"], dataset_attributes_path = config["dataset_attributes_path"] ,\
                pretrain_feature_path = config["pretrain_feature_path"])

        # #find consine accuracy
        # load_dir = config["test_folder_path"] + "cos_dict.npy"
        # cos_dict = np.load(load_dir)
        
        # # game_max_coss = np.array(cos_dict.item().get("game_max_coss"))
        # game_mean_coss = np.array(cos_dict.item().get("game_mean_coss"))
        # test_accr = np.array(cos_dict.item().get("test_accr"))

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
    

        # game.evaluate(100000,"../../../Desktop/Datasets/Number_Set/Number_Set_7dis_12attr_5max_Test.npy" , \
        #     test_permute = False, dir_name = "checkpoints_train_NS_7_cand_without_permute_no_pretrain/",\
        #         ckpt_name="informed-100001", dataset_attributes_path = None,\
        #         pretrain_feature_path = None)
        
     
        # sess.close()
    
if __name__ == "__main__":
    main()
