# Copyright (c) 2020 Complicateddd Authors. All Rights Reserved.


import os
basedir = os.path.abspath(os.path.dirname(__file__))

class hyper_params(object):


    def __init__(self,):

        self.use_gpu=True
        self.ir_optim=True
        self.use_tensorrt=False
        self.use_fp16=False
        self.gpu_mem=8000
        self.rec_char_type='ch'

        self.det_algorithm='DB'

        self.det_model_dir=str(basedir)+"/weights/ch_ppocr_mobile_v2.0_det_infer"

        self.det_limit_side_len=960
        self.det_limit_type='max'

        self.det_db_thresh=0.3
        self.det_db_box_thresh=0.5
        self.det_db_unclip_ratio=1.6
        self.max_batch_size=10

        self.det_east_score_thresh=0.8
        self.det_east_cover_thresh=0.1
        self.det_east_nms_thresh=0.2

        self.det_sast_score_thresh=0.5
        self.det_sast_nms_thresh=0.2
        self.det_sast_polygon=False

        self.rec_algorithm='CRNN'

        self.rec_model_dir=str(basedir)+"/weights/ch_ppocr_mobile_v2.0_rec_infer" 

        self.rec_image_shape="3, 32, 320"
        self.rec_batch_num=1
        self.max_text_length=25
        self.rec_char_dict_path=str(basedir)+"/ppocr/utils/ppocr_keys_v1.txt"
        self.use_space_char=True

        self.vis_font_path="./doc/simfang.ttf"
        self.drop_score=0.5

        self.use_angle_cls=True

        self.cls_model_dir=str(basedir)+"/weights/ch_ppocr_mobile_v2.0_cls_infer"

        self.cls_image_shape="3, 48, 192"
        self.label_list=['0', '180']
        self.cls_batch_num=6
        self.cls_thresh=0.9

        self.enable_mkldnn=False
        self.use_zero_copy_run=False
        self.use_pdserving=False
