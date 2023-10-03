# 结论：当模型进行ensemble的时候性能会明显提升。
import json
import numpy as np
import os, sys
from collections import Counter
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import torch
from random import choice
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats import inter_rater as irr

valid_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31]
valid_joints_num = len(valid_joints)
joint_dim = 4

# 读取json文件
def sp_json(file_path):
    load_dict = json.load(open(file_path, 'r', encoding='utf-8'))
    frame_data = load_dict['frames']
    frame_num = len(frame_data)
    joint_data = [0 for x in range(frame_num)]
    for i in range(frame_num):
        frame_data_i = frame_data[i]
        body_data_i = frame_data_i['bodies']
        if len(body_data_i) == 1:
            body_data_i = body_data_i[0]
            joint_data_i = body_data_i['joint_orientations']
            joint_data_i = [joint_data_i[j] for j in valid_joints]
        else:
            joint_data_i = [0 for x in range(joint_dim * valid_joints_num)]
        joint_data[i] = np.array(joint_data_i).flatten().tolist()
    return joint_data


if __name__ == '__main__':
    front_root_path = "../../data/skeleton_data/front"
    score_root_path = "../../data/expert_score"
    sub_path = os.listdir(front_root_path)
    file_num = len(sub_path)
    front_torch_joints = []
    evaluate_labels = []
    class_labels = []
    num_of_sample_frames = []
    classes_mapping = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4, 10: 5, 11: 6, 12: 6, 13: 6, 14: 6}
    json_scores = json.load(open(os.path.join(score_root_path, "experts_score.json")))
    experts_score_list = []
    for i in range(file_num):
        short_path = sub_path[i]
        front_file_path = os.path.join(front_root_path, short_path)
        front_load_list = sp_json(front_file_path)
        front_load_list_scale = preprocessing.scale(front_load_list)

        subject_index = short_path[short_path.index('_s') + 1:short_path.index('_s') + 4]
        action_index = short_path[short_path.index('_m') + 1:short_path.index('_m') + 4]
        episode_index = short_path[short_path.index('_e') + 1:short_path.index('_e') + 3]
        experts_score = json_scores[subject_index][action_index][episode_index]
        experts_score_list.append(experts_score)
        experts_score_count = Counter(experts_score)
        # most common element in experts_score_count
        most_common_score = experts_score_count.most_common(1)[0]
        if (most_common_score[1] == 1):
            continue
        else:
            final_score = most_common_score[0]
            front_torch_joints.append(torch.tensor(front_load_list_scale, dtype=torch.float))
            evaluate_labels.append(final_score - 1)
            class_labels.append(classes_mapping[int(action_index[1:]) - 1])
            num_of_sample_frames.append(front_load_list_scale.shape[0])
    labels = torch.tensor(evaluate_labels, dtype=torch.int64)
    class_labels = torch.tensor(class_labels, dtype=torch.int64)
    num_of_sample_frames = torch.tensor(num_of_sample_frames, dtype=torch.int64)
    experts_score_array = np.array(experts_score_list)
    #np.savez("annotation_data.npz",experts_score_array, class_labels)
    fk = irr.fleiss_kappa(irr.aggregate_raters(experts_score_array)[0], method='fleiss')
    print(fk)
    for i in range(7):
        class_i_indexes = torch.where(class_labels == i)
        class_i_labels = labels[class_i_indexes]
        class_i_labels = class_i_labels.tolist()
        class_i_num_of_frames = num_of_sample_frames[class_i_indexes]
        class_i_num_of_frames = class_i_num_of_frames.tolist()
        print(Counter(class_i_labels))
        print(np.mean(np.array(class_i_num_of_frames)))
        print(np.max(np.array(class_i_num_of_frames)))
        #print(class_i_num_of_frames)

