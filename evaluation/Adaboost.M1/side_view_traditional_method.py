#结论：当模型进行ensemble的时候性能会明显提升。
import json
import numpy as np
import os, sys
from collections import Counter
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import cohen_kappa_score
from logger import init_logger
log_file = "log/side_view_traditional_method.txt"
logger = init_logger(log_file)

valid_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
valid_joints_num = len(valid_joints)
joint_dim = 4
#读取json文件
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
            joint_data_i = [0 for x in range(joint_dim*valid_joints_num)]
        joint_data[i] = np.array(joint_data_i).flatten().tolist()
    return joint_data


if __name__ == '__main__':
    front_root_path = "../../data/skeleton_data/front"
    side_root_path = "../../data/skeleton_data/side"
    score_root_path = "../../data/expert_score"
    sub_path = os.listdir(front_root_path)
    file_num = len(sub_path)
    front_torch_joints = []
    side_torch_joints = []
    evaluate_labels = []
    class_labels = []
    NUM_OF_TESTS = 10
    classes_mapping = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4, 10: 5, 11: 6, 12: 6, 13: 6, 14: 6}
    json_scores = json.load(open(os.path.join(score_root_path, "experts_score.json")))
    for i in range(file_num):
        short_path = sub_path[i]
        front_file_path = os.path.join(front_root_path, short_path)
        front_load_list = sp_json(front_file_path)
        front_load_list_scale = preprocessing.scale(front_load_list)
        side_file_path = front_file_path.replace('front', 'side')
        #import pdb; pdb.set_trace()
        side_load_list = sp_json(side_file_path)
        side_load_list_scale = preprocessing.scale(side_load_list)
        
        subject_index = short_path[short_path.index('_s')+1:short_path.index('_s')+4]
        action_index = short_path[short_path.index('_m')+1:short_path.index('_m')+4]
        episode_index = short_path[short_path.index('_e')+1:short_path.index('_e')+3]
        experts_score = json_scores[subject_index][action_index][episode_index]
        experts_score_count = Counter(experts_score)
        # most common element in experts_score_count
        most_common_score = experts_score_count.most_common(1)[0]
        if(most_common_score[1] == 1):
            continue
        else:
            final_score = most_common_score[0]
            front_torch_joints.append(torch.tensor(front_load_list_scale, dtype=torch.float))
            side_torch_joints.append(torch.tensor(side_load_list_scale, dtype=torch.float))
            evaluate_labels.append(final_score - 1)
            class_labels.append(classes_mapping[int(action_index[1:])-1]) 
    data = rnn_utils.pad_sequence(side_torch_joints, batch_first=True, padding_value = 0)
    labels = torch.tensor(evaluate_labels, dtype=torch.int64)
    class_labels = torch.tensor(class_labels, dtype=torch.int64)
    final_micro_f1_results_by_classes = {f'class_{index}': [] for index in range(7)}
    final_macro_f1_results_by_classes = {f'class_{index}': [] for index in range(7)}
    final_kappa_results_by_classes = {f'class_{index}': [] for index in range(7)}
    
    for i in range(NUM_OF_TESTS):
        x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(data, labels, class_labels, test_size=0.3, random_state=0)
        new_x_train = np.reshape(x_train, [x_train.shape[0], -1])
        new_x_test = np.reshape(x_test, [x_test.shape[0], -1])
        ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), learning_rate=1.0, n_estimators=20)
        logger.info('begin to train')
        ada.fit(new_x_train, y_train)
        logger.info('begin to test')
        y_pred = ada.predict(new_x_test)
        logger.info('compute the metrics')
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        kappa_score = cohen_kappa_score(y_test, y_pred)
        logger.info('micro f1: {:0.3f}, macro f1: {:0.3f}, kappa: {:0.3f}'.format(micro_f1, macro_f1, kappa_score))

        
        # 按照类别测试
        num_classes = 7
        logger.info('In test phase for each action...')
        true_labels_tensor = torch.tensor(y_test)
        predict_labels_tensor = torch.tensor(y_pred)
        for c in range(num_classes):
            test_indexes = torch.where(z_test == c)[0]
            true_labels_c = true_labels_tensor[test_indexes.tolist()]
            predict_labels_c = predict_labels_tensor[test_indexes.tolist()]
            micro_f1 = f1_score(true_labels_c, predict_labels_c, average='micro')
            macro_f1 = f1_score(true_labels_c, predict_labels_c, average='macro')
            kappa_score = cohen_kappa_score(true_labels_c, predict_labels_c)
            final_micro_f1_results_by_classes[f'class_{c}'].append(micro_f1)
            final_macro_f1_results_by_classes[f'class_{c}'].append(macro_f1)
            final_kappa_results_by_classes[f'class_{c}'].append(kappa_score)
            
            logger.info('Class i: {:d}, Micro F1: {:0.3f}, macro F1: {:0.3f}, kappa: {:0.3f}'.format(c, micro_f1, macro_f1, kappa_score))
    
