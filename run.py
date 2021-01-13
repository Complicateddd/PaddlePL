# Copyright (c) 2020 Complicateddd Authors. All Rights Reserved.

import os
import sys

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import cv2
import copy
import numpy as np
import time
from PIL import Image
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppocr.utils.logging import get_logger
from tools.infer.utility import draw_ocr_box_txt
from help_filter import RotateAntiClockWise90,fine_colone_text,find_box_range

class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        if dt_boxes is None:
            return None, None
        img_crop_list = []
        dt_boxes = sorted_boxes(dt_boxes)
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            
        rec_res, elapse = self.text_recognizer(img_crop_list)
       
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        return filter_boxes, filter_rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def main(path_img_test, path_submit):
    from hyper_config import hyper_params
    args=hyper_params()
    image_file_list,sigle_lists = get_image_file_list(path_img_test)
    text_sys = TextSystem(args)
    # is_visualize = True
    font_path = args.vis_font_path
    drop_score = args.drop_score

    down_result_pred=['' for _ in range(len(sigle_lists))]

    for index,image_file in enumerate(image_file_list):
        
        img = cv2.imread(image_file)
        if img is None:
            continue
        height,width=img.shape[0],img.shape[1]
        if height>width:
            img=RotateAntiClockWise90(img)
            height,width=img.shape[0],img.shape[1]

        limit_range_val_width=width*0.045
        limit_range_val_height=height*0.013

        starttime = time.time()
        dt_boxes, rec_res = text_sys(img)
        if not rec_res:
            continue
        elapse = time.time() - starttime
        xiangmu_name,jieguo_name,danwei_name=find_box_range(dt_boxes,rec_res)

        for xiangmu_name_id in range(len(xiangmu_name)):

            xiangmumingcheng_colomn_list,xiangmumingcheng_box_colomn_list=fine_colone_text(dt_boxes,xiangmu_name[xiangmu_name_id],rec_res,limit_range=limit_range_val_width)
            jieguo_colomn_list,jieguo_box_colomn_list=fine_colone_text(dt_boxes,jieguo_name[xiangmu_name_id],rec_res,limit_range=limit_range_val_width/2)
            danwei_colomn_list,danwei_box_colomn_list=fine_colone_text(dt_boxes,danwei_name[xiangmu_name_id],rec_res,limit_range=limit_range_val_width/1.5)

            ans=[]
            ######  if match correctly :
            if len(xiangmumingcheng_colomn_list)==len(jieguo_colomn_list)==len(danwei_colomn_list):
                for i in range(len(jieguo_colomn_list)):
                    ele_set=['','','']
                    ele_set[0]=xiangmumingcheng_colomn_list[i]
                    ele_set[1]=jieguo_colomn_list[i]
                    ele_set[2]=danwei_colomn_list[i]
                    ans.append(ele_set)
            else:
                for i in range(len(jieguo_colomn_list)):
                    ele_set=['','','']
                    ele_set[1]=jieguo_colomn_list[i]
                    for name_id in range(len(xiangmumingcheng_colomn_list)):
                        if abs(xiangmumingcheng_box_colomn_list[name_id][3][1]-jieguo_box_colomn_list[i][3][1])<=limit_range_val_height:
                            ele_set[0]=xiangmumingcheng_colomn_list[name_id].replace('*', '')
                            xiangmumingcheng_colomn_list.pop(name_id)
                            xiangmumingcheng_box_colomn_list.pop(name_id)
                            break
                    for name_id in range(len(danwei_colomn_list)):
                        if abs(danwei_box_colomn_list[name_id][3][1]-jieguo_box_colomn_list[i][3][1])<=limit_range_val_height:
                            ele_set[2]=danwei_colomn_list[name_id]
                            danwei_colomn_list.pop(name_id)
                            danwei_box_colomn_list.pop(name_id)
                            break
                    ans.append(ele_set)
            ans=[''.join(ele) for ele in ans]
            down_result=''
            for ele in ans:
                down_result+=ele

            down_result_pred[index]+=down_result


    submit = pd.DataFrame({'id': sigle_lists, 'content': down_result_pred})
    
    submit.to_csv(path_submit, index=None, encoding='utf-8')


if __name__ == "__main__":
    path_img_test = sys.argv[1]
    path_submit = sys.argv[2]
    main(path_img_test, path_submit)


