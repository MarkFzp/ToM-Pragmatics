import tensorflow as tf
import numpy as np
import os
from Game import Game
from easydict import EasyDict as edict

USE_GPU = True
configuration = None
# config = None
if not USE_GPU:
    # [for CPU training]
    # config = tf.ConfigProto(device_count = {'GPU': 0})
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    configuration = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    configuration.gpu_options.allow_growth = True

tf.reset_default_graph()
# tf.set_random_seed(32)
# np.random.seed(32)
# np.set_printoptions(precision=2, threshold=np.nan)

def random_generated():
    import sys
    # sys.path.insert(1, '../CL_Net/Referential_Game/Number_Set')
    # from concept import Concept
    from easydict import EasyDict as edict
    

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
    num_attributes = alphabet_size = config["num_attributes"]
    input_data_len = config["input_data_len"]
    target_size = 10000
    num_candidates = config["num_candidates"]
    num_epoches = 1

    dense_len = 50
    message_len = 1
    batch_size = 100
    iteration = 100000
    learning_rate = 1e-3
    beta_listener = 0.001
    beta_speaker = 0.01
   
    temperature = 0.01

    
    sess = tf.Session(config = configuration)
    
    # print(type(config["dataset_attributes_path"]))
    # exit()
    game = Game(sess, 'fc', input_data_len, batch_size, num_epoches, max_concept_len, num_candidates, \
        num_attributes, dense_len, message_len, alphabet_size, learning_rate=learning_rate,\
             beta_listener=beta_listener, beta_speaker=beta_speaker, temperature=temperature,\
            concept_folder = config["concept_folder"], train_permutation=config["train_permute"], test_permutation = config["test_permute"],\
            train_set_path = config["train_dataset_path"],\
                test_set_path = config["test_dataset_path"])
            
    game.train(iteration, "checkpoint", config["test_folder_path"], pretrain = config["pretrain"], \
        dataset_attributes_path = config["dataset_attributes_path"],\
        pretrain_feature_path = config["pretrain_feature_path"])

    game.evaluate(100000, config['test_folder_path'],config['ckpt_name'], \
        dataset_attributes_path = config["dataset_attributes_path"],\
        pretrain_feature_path = config["pretrain_feature_path"]
    )

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
    
    #test hard problem

    # print("hard problem error rate: ")
    # game_hard = Game(sess_hard, 'fc', hard_data, hard_target_indices, dense_len, message_len, alphabet_size, learning_rate=learning_rate, beta_listener=beta_listener, beta_speaker=beta_speaker, temperature=temperature)
    # game.evaluate(hard_data, hard_target_indices, "rand_atanh_dl10_ml1_temp0.35_bl0.01_bs0.01")

    #test rtd = 1 problem
    # print("rtd = 1 problem error rate: ")
    # game_rtd = Game(sess_rtd, 'fc', rtd_data, rtd_target_indices, dense_len, message_len, alphabet_size, learning_rate=learning_rate, beta_listener=beta_listener, beta_speaker=beta_speaker, temperature=temperature)
    # game.evaluate(rtd_data, rtd_target_indices, "rand_atanh_dl10_ml1_temp0.35_bl0.01_bs0.01")

    #test rtd = 1 hard problem
    # print("rtd=1 hard problem error rate: ")

    # game_hard_rtd = Game(sess_hard_rtd, 'fc', rtd_hard_data, rtd_hard_target_indices, dense_len, message_len, alphabet_size, learning_rate=learning_rate, beta_listener=beta_listener, beta_speaker=beta_speaker, temperature=temperature)
    # game.evaluate(rtd_hard_data, rtd_hard_target_indices, "rand_atanh_dl10_ml1_temp0.35_bl0.01_bs0.01")

    sess.close()


def visa(file_name):
    import json
    from random import shuffle
    from tensorflow.keras.utils import to_categorical as onehot

    objects = []
    attributes = set()
    data = []
    with open(file_name) as f:
        dict_data = json.load(f)
        print(dict_data)
        for obj, attrib in dict_data.items():
            attributes.update(attrib)
            objects.append(obj)
        attributes = list(attributes)
        shuffle(attributes)
        shuffle(objects)
        for obj in objects:
            attrib_indices = np.zeros([len(attributes), ], dtype=np.int32)
            for attrib in dict_data[obj]:
                attrib_indices[attributes.index(attrib)] = 1
            data.append(attrib_indices)
    print(len(attributes))
    data = np.array(data)






if __name__ == "__main__":
    random_generated()
    # visa('../../Jupyter/visa.json')
