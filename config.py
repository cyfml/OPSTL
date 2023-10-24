import os
from sacred import Experiment

ex = Experiment("PSTL", save_git_info=False) 

@ex.config
def my_config():
    ############################## setting ##############################
    version = "ntu60_xsub_joint"
    dataset = "NTU60_occ"   # ntu60 / ntu120 
    split = "xsub"
    view = "joint"      # joint / motion / bone
    save_lp = False
    save_finetune = False
    save_semi = False
    pretrain_epoch = 150
    ft_epoch = 150
    lp_epoch = 150
    pretrain_lr = 5e-3
    lp_lr = 0.01
    ft_lr = 5e-3
    label_percent = 0.1
    weight_decay = 1e-5
    hidden_size = 256
    label_num = 60
    ############################## ST-GCN ###############################
    in_channels = 3
    hidden_channels = 16
    hidden_dim = 256
    dropout = 0.5
    graph_args = {
    "layout" : 'ntu-rgb+d',
    "strategy" : 'spatial'
    }
    edge_importance_weighting = True
    ############################ down stream ############################
    weight_path = '/cvhci/temp/ychen2/data_occ_frame50/OPSTL/kmeans+knn/OPSTL_'+version+'_frame50_epoch_150_pretrain.pt'
    train_mode = 'pretrain'
    # train_mode = 'finetune'
    # train_mode = 'pretrain'
    # train_mode = 'semi'
    log_path = '/cvhci/temp/ychen2/data_occ_frame50/OPSTL/kmeans+knn/'+version+'_'+train_mode+'.log'
    ################################ GPU ################################
    gpus = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    ########################## Skeleton Setting #########################
    batch_size = 128
    channel_num = 3
    person_num = 2
    joint_num = 25
    max_frame = 50
    # train_list = '/cvhci/temp/ychen2/data_occ_frame50/'+dataset+'_frame50_relative/'+split+'/train_position.npy'
    # test_list = '/cvhci/temp/ychen2/data_occ_frame50/'+dataset+'_frame50_relative/'+split+'/val_position.npy'
    train_list = '/cvhci/temp/ychen2/data_occ_frame50/OPSTL/kmeans+knn/'+dataset+'_completed_frame50/'+split+'/train_position.npy'
    test_list = '/cvhci/temp/ychen2/data_occ_frame50/OPSTL/kmeans+knn/'+dataset+'_completed_frame50/'+split+'/val_position.npy'
    train_label = '/lsdf/users/ychen/data_occ/NTU-RGB-D-60-occ/'+split+'/train_label.pkl'
    test_label = '/lsdf/users/ychen/data_occ/NTU-RGB-D-60-occ/'+split+'/val_label.pkl'
    joints_list = '/lsdf/users/ychen/data_occ/NTU-RGB-D-60-occ/'+split+'/train_missing_joints_distribution.pkl'
    ########################### complete joints ###########################
    original_train_data_nan = '/lsdf/users/ychen/data_occ/NTU-RGB-D-60-occ/'+split+'/train_data_nan.npy'
    original_test_data_nan = '/lsdf/users/ychen/data_occ/NTU-RGB-D-60-occ/'+split+'/val_data_nan.npy'
    missing_joints_train = '/lsdf/users/ychen/data_occ/NTU-RGB-D-60-occ/'+split+'/train_missing_joints.pkl'
    missing_joints_test = '/lsdf/users/ychen/data_occ/NTU-RGB-D-60-occ/'+split+'/val_missing_joints.pkl'
    output_path = '/cvhci/temp/ychen2/data_occ_frame50/OPSTL/kmeans+knn'
    cluster_num = 60
    n_neighbors = 5
    ########################### Data Augmentation #########################
    temperal_padding_ratio = 6
    shear_amp = 1
    mask_joint = 9
    mask_frame = 10
    ############################ Barlow Twins #############################
    pj_size = 6144
    lambd = 2e-4