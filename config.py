#!/usr/bin/env python
# coding=utf-8
import os


class Configuration(object):
    def __init__(self):
        gpu_no = 0
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_no)

        # make_sample conf
        self.interval = 1
        self.stride = 1

        # model conf
        self.model_mode = '2d'
        self.class_num = 2
        self.class_weights = (1, 1)
        self.loss_type = 'cross_entropy'
        self.is_training = True
        self.reuse = False

        # operationer conf
        self.log_path = 'C:/users/12638/dfsdata2/zhangyao_data/DB/SpineSagT2W/standard_data/log/'
        self.model_path = '.\\trained\\'
        self.prediction_path = 'C:/users/12638/dfsdata2/zhangyao_data/DB/SpineSagT2W/standard_data/prediction/'
        self.need_restore = True

        self.postprocess = True

        self.learning_rate = 0.0001
        self.batch_size = 1
        self.epochs = 50
        self.display_step = 500
        self.optimizer = 'adam'

        # data_provider conf
        self.data_patch_dir = 'C:/users/12638/dfsdata2/zhangyao_data/DB/SpineSagT2W/standard_data/train_data/data_patch/slice' + str(self.interval) + '/'
        self.seg_patch_dir = 'C:/users/12638/dfsdata2/zhangyao_data/DB/SpineSagT2W/standard_data/train_data/seg_patch/slice' + str(self.interval) + '/'

        self.data_volume_dir = 'C:/users/12638/dfsdata2/zhangyao_data/DB/SpineSagT2W/standard_data/data_volume/'
        self.seg_volume_dir = 'C:/users/12638/dfsdata2/zhangyao_data/DB/SpineSagT2W/standard_data/seg_volume/'

        self.val_case_id_list = [x[:-4] for x in os.listdir(self.data_volume_dir) if int(x[4:-4]) in range(196, 211)]
        self.train_case_id_list = [x[:-4] for x in os.listdir(self.data_volume_dir) if int(x[4:-4]) not in range(196, 211)]

