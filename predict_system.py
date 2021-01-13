# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

import warnings
warnings.filterwarnings('ignore')


__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

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

logger = get_logger()


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
            logger.info(bno, rec_res[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        logger.info("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
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
            logger.info("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        logger.info("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
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


def find_box_range(box,rec_result):
    ## find ans the box range for '项目名称'，‘结果’，‘单位’
    # box_range_list=[[[0. ,0.],[0., 0.],[0. ,0.],[0., 0.]],
    # [[0., 0.],[0. ,0.],[0. ,0.],[0.,0.]],[[0., 0.],[0. ,0.],[0., 0.],[0., 0.]]]

    xiangmu_name=[]
    jieguo_name=[]
    danwei_name=[]


    for i,(text,score) in enumerate(rec_result):
        if "项目名称" in text:
            xiangmu_name.append(box[i])
        elif '结果' in text:
            jieguo_name.append(box[i])
        elif '单位' in text:
            danwei_name.append(box[i])
    min_length=min(len(xiangmu_name),len(jieguo_name),len(danwei_name))
    # assert len(box_range_list)==3
    return xiangmu_name[:min_length],jieguo_name[:min_length],danwei_name[:min_length]




def fine_colone_text(box,box_range_list_1,rec_result,limit_range=5):
    ''' find '项目名称' colomn text
    return list len: len(colomn text)

    '''
    x_left_max=min(box_range_list_1[0][0],box_range_list_1[3][0])
    x_right_max=max(box_range_list_1[1][0],box_range_list_1[2][0])

    x_left_limit=x_left_max-limit_range
    x_right_limit=x_right_max+limit_range

    colomn_list=[]
    colomn_cor_list=[]

    colomn_y_cor=[]
    for j,cord in enumerate(box):
        if min(cord[0][0],cord[3][0])>=x_left_limit and max(cord[1][0],cord[2][0])<=x_right_limit:
            colomn_list.append(rec_result[j][0])
            colomn_cor_list.append(cord)
            colomn_y_cor.append(cord[0][1])

    colomn_y_cor_sorted_index=np.argsort(np.array(colomn_y_cor))
    colomn_list=np.array(colomn_list)[colomn_y_cor_sorted_index].tolist()
    colomn_cor_list=np.array(colomn_cor_list)[colomn_y_cor_sorted_index].tolist()

    return colomn_list,colomn_cor_list




def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    drop_score = args.drop_score
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        # print(img.shape)
        height,width=img.shape[0],img.shape[1]

        limit_range_val_width=width*0.05
        limit_range_val_height=height*0.013

        starttime = time.time()
        dt_boxes, rec_res = text_sys(img)
        elapse = time.time() - starttime
        logger.info("Predict time of %s: %.3fs" % (image_file, elapse))

        # for i,(text, score) in enumerate(rec_res):
        #     if text=='项目名称':
        #         logger.info("{}, {:.3f}".format(text, score))
        #         print(dt_boxes[i])
        xiangmu_name,jieguo_name,danwei_name=find_box_range(dt_boxes,rec_res)


        for xiangmu_name_id in range(len(xiangmu_name)):

        # box_range_list=find_box_range(dt_boxes,rec_res)

            xiangmumingcheng_colomn_list,xiangmumingcheng_box_colomn_list=fine_colone_text(dt_boxes,xiangmu_name[xiangmu_name_id],rec_res,limit_range=limit_range_val_width)
            # print(xiangmumingcheng_colomn_list)

            jieguo_colomn_list,jieguo_box_colomn_list=fine_colone_text(dt_boxes,jieguo_name[xiangmu_name_id],rec_res,limit_range=limit_range_val_width/2)

            danwei_colomn_list,danwei_box_colomn_list=fine_colone_text(dt_boxes,danwei_name[xiangmu_name_id],rec_res,limit_range=limit_range_val_width/1.5)
            # print(xiangmumingcheng_colomn_list)
            # print(jieguo_colomn_list)
            # print(danwei_colomn_list)


            ans=[]

            for i in range(len(jieguo_colomn_list)):
                ele_set=['','','%']
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
                down_result+=(ele+'/')
            print(down_result)

        # print(box_range_list)
        # for i,(text, score) in enumerate(rec_res):
        #     if text=='项目名称':
        #         logger.info("{}, {:.3f}".format(text, score))
        #         print(dt_boxes[i])

        if is_visualize:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            # print(rec_res)
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]

            draw_img = draw_ocr_box_txt(
                image,
                boxes,
                txts,
                scores,
                drop_score=drop_score,
                font_path=font_path)
            draw_img_save = "./inference_results/"
            if not os.path.exists(draw_img_save):
                os.makedirs(draw_img_save)
            cv2.imwrite(
                os.path.join(draw_img_save, os.path.basename(image_file)),
                draw_img[:, :, ::-1])
            logger.info("The visualized image saved in {}".format(
                os.path.join(draw_img_save, os.path.basename(image_file))))


if __name__ == "__main__":



    from hyper_config import hyper_params
        args=hyper_params()

    path_img_test = sys.argv[1]
    path_submit = sys.argv[2]

    main(path_img_test, path_submit)
    

    # main(utility.parse_args())
    # main(arg)


