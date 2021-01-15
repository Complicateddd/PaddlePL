import cv2
import numpy as np

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

    return colomn_list[1:],colomn_cor_list[1:]

def RotateAntiClockWise90(img):
    trans_img = cv2.transpose(img)
    new_img = cv2.flip(trans_img, 0)
    return new_img