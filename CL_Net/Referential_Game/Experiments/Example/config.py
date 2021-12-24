from easydict import EasyDict as edict
import numpy as np
np.set_printoptions(suppress = True, precision = 4)

train_config = edict()
train_config.device = "/cpu:0"
train_config.batch_size = 128
train_config.feature_lr = 1e-3
train_config.belief_lr = 1e-4
train_config.feature_decay = 2e4
train_config.belief_decay = None
train_config.regularization_param = 0
train_config.iteration_phase1 = 8e4
train_config.iteration_phase2 = 6e4
train_config.iteration_phase3 = 4e4
train_config.iteration_phase4 = 4e4
train_config.continue_steps = 0

game_config = edict()
game_config.num_distractors = 4
game_config.rl_gamma = 0.6
game_config.rational_beta = 20
game_config.dataset = 'Number_Set' #Number_Set, ImageNet, Geometry

vision_config = edict()
if game_config.dataset == 'Geometry':
    vision_config.type = 'CNN' #CNN, VGG16, Feat
    game_config.message_space_size = 18
    game_config.attributes = range(18)
    game_config.n_grid_per_side = 2
    game_config.img_h = 128
    game_config.img_w = 128
    game_config.dataset_attributes_path = '/home/Datasets/Geometry/Geometry_attributes.npy'
elif game_config.dataset == 'Number_Set':
    vision_config.type = 'Feat' #CNN, VGG16, Feat
    game_config.message_space_size = 10
    game_config.attributes = range(10)
    game_config.concept_max_size = 4
    game_config.permutation = False
    game_config.generated_dataset = True
    game_config.generated_dataset_path = '/home/Datasets/Number_Set/Number_Set_4dis_Train.npy'
elif game_config.dataset == 'ImageNet':
    vision_config.type = 'VGG16' #CNN, VGG16, Feat
    game_config.message_space_size = 25
    game_config.attributes = range(25)
    game_config.img_h = 224
    game_config.img_w = 224
    game_config.bbox_ratio_thred = 0.5
    game_config.dataset_attributes_path = '/home/Datasets/ImageNet/ImageNet_attributes.npy'

#can overwrite default vision_config here
vision_config.type = 'Feat'
if vision_config.type == 'CNN':
    vision_config.img_h = 128
    vision_config.img_w = 128
    vision_config.cnn_configs = [(7, 2, 128, False), (7, 2, 64, True), (5, 2, 32, True)]
    vision_config.fc_configs = [36, 36]
    game_config.images_path = '/home/Datasets/%s/%s_images.npy' % (game_config.dataset, game_config.dataset)
elif vision_config.type == 'Feat':
    if game_config.dataset == 'ImageNet':
        vision_config.feat_dim = 4096
    elif game_config.dataset == 'Geometry':
        vision_config.feat_dim = 36
    elif game_config.dataset == 'Number_Set':
        vision_config.feat_dim = game_config.message_space_size
    '''
    if game_config.feature_dir is None or not existing: load from config path
    we don't overwrite to /home/Datasets, please copy working features to /home/Datasets directly
    '''
    # game_config.feat_dir = '/home/Datasets/%s/%s_feat_%d' % (game_config.dataset, game_config.dataset, game_config.num_distractors)
    game_config.feat_dir = None
elif vision_config.type == 'VGG16':
    vision_config.img_h = 224
    vision_config.img_w = 224
    vision_config.vgg16_npy_path = '/home/Datasets/vgg16.npy'
    game_config.images_path = '/home/Datasets/%s/%s_images.npy' % (game_config.dataset, game_config.dataset)
else:
    print('Wrong vision type')
    assert(0)

belief_config = edict()
belief_config.type = 'OrderFree'
belief_config.fc_configs = [3 * game_config.message_space_size]
if belief_config.type != 'OrderFree':
    belief_config.num_dims = [game_config.num_distractors, game_config.num_distractors]
    belief_config.num_filters = [2 * game_config.message_space_size, game_config.message_space_size]
else:
    belief_config.context_length = [int(0.8 * game_config.message_space_size),
                                    int(0.6 * game_config.message_space_size)]
    belief_config.single_length = [2 * game_config.message_space_size, game_config.message_space_size]
belief_config.belief_var_1d = -50
belief_config.batch_norm = False

if belief_config.type == 'Explicit' or 'OrderFree':
    belief_config.boltzman_beta = 1
elif belief_config.type == 'Implicit':
    belief_config.message_space_size = game_config.message_space_size
else:
    print('Wrong belief type')
    assert(0)

value_config = edict()
value_config.fc_layer_info_ = [3 * game_config.message_space_size, 3 * game_config.message_space_size]
value_config.cnn_layer_info_ = [(2 * game_config.message_space_size, game_config.num_distractors),
                                (game_config.message_space_size, game_config.num_distractors),
                                (1, game_config.num_distractors)]
value_config.null_belief_penalty = -1
