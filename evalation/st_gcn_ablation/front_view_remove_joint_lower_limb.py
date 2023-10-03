# 均匀采样效果较差
import json
import numpy as np
import os, sys
import random
from collections import Counter
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from random import choice
from sklearn.metrics import cohen_kappa_score
from logger import init_logger
from st_gcn_part_remove_joint_leg import *
from sklearn.metrics import cohen_kappa_score

#(N, in_channels, T_{in}, V_{in}, M_{in})
log_file = "log/front_view_remove_joint_leg.txt"
logger = init_logger(log_file)

valid_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 26, 27, 28, 29, 30, 31]
valid_joints_num = len(valid_joints)
joint_dim = 3

#读取json文件
def sp_json(file_path):
    load_dict = json.load(open(file_path, 'r', encoding='utf-8'))
    frame_data = load_dict['frames']
    frame_num = len(frame_data)
    joint_data = []
    for i in range(frame_num):
        frame_data_i = frame_data[i] 
        body_data_i = frame_data_i['bodies']
        if len(body_data_i) == 1:
            body_data_i = body_data_i[0]
            joint_data_i = body_data_i['joint_positions']
            joint_data_i = [joint_data_i[j] for j in valid_joints]
            joint_data_i = sum(joint_data_i, [])
        else:
            joint_data_i = [0 for x in range(joint_dim*valid_joints_num)]
        joint_data.append(joint_data_i)
    return joint_data

def get_samples_indexes(arr_len, samples_amount):
    interval = float(arr_len) / samples_amount
    samples_indexes = list()
    for index in range(samples_amount):
        samples_indexes.append(min(round(index * interval), arr_len - 1))
    return samples_indexes

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

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
    BATCH_SIZE = 32
    NUM_EPOCHS = 500
    learning_rate=0.01  #学习率
    NUM_OF_TESTS = 5
    device = 'cuda:0'
    classes_mapping = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4, 10: 5, 11: 6, 12: 6, 13: 6, 14: 6}
    json_scores = json.load(open(os.path.join(score_root_path, "experts_score.json")))
    for i in range(file_num):
        short_path = sub_path[i]
        front_file_path = os.path.join(front_root_path, short_path)
        front_load_list = sp_json(front_file_path)
        side_file_path = front_file_path.replace('front', 'side')
        side_load_list = sp_json(side_file_path)
        
        subject_index = short_path[short_path.index('_s')+1:short_path.index('_s')+4]
        action_index = short_path[short_path.index('_m')+1:short_path.index('_m')+4]
        episode_index = short_path[short_path.index('_e')+1:short_path.index('_e')+3]
        experts_score = json_scores[subject_index][action_index][episode_index]
        experts_score_count = Counter(experts_score)
        # most common element in experts_score_count
        most_common_score = experts_score_count.most_common(1)[0]
        #import pdb;
        #pdb.set_trace();
        if(most_common_score[1] == 1):
            continue
        else:
            final_score = most_common_score[0]
            front_torch_joints.append(torch.tensor(front_load_list, dtype=torch.float))
            side_torch_joints.append(torch.tensor(side_load_list, dtype=torch.float))
            evaluate_labels.append(final_score - 1)
            class_labels.append(classes_mapping[int(action_index[1:])-1]) 
    front_data = rnn_utils.pad_sequence(front_torch_joints, batch_first=True, padding_value=0)
    front_data = front_data.reshape([front_data.shape[0], front_data.shape[1], valid_joints_num, joint_dim, 1])
    side_data = rnn_utils.pad_sequence(side_torch_joints, batch_first=True, padding_value=0)
    side_data = side_data.reshape([side_data.shape[0], side_data.shape[1], valid_joints_num, joint_dim, 1])
    #data = torch.cat((front_data, side_data), axis = 3)

    data = front_data
    num_class = 3
    graph_args = {'strategy': 'spatial', 'layout': 'azure'}

    #data = data.reshape([data.shape[0], data.shape[1], valid_joints_num, 2*joint_dim, 1])
    data = data.permute([0, 3, 1, 2, 4])
    labels = torch.tensor(evaluate_labels, dtype=torch.int64)
    class_labels = torch.tensor(class_labels, dtype=torch.int64)
    final_micro_f1_results = []
    final_macro_f1_results = []
    final_kappa_results = []
    final_micro_f1_results_by_classes = {f'class_{index}': [] for index in range(7)}
    final_macro_f1_results_by_classes = {f'class_{index}': [] for index in range(7)}
    final_kappa_results_by_classes = {f'class_{index}': [] for index in range(7)}
    total_test_acc_0 = []
    total_test_acc_1 = []
    
    for i in range(NUM_OF_TESTS):
        x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(data, labels, class_labels, test_size=0.3, random_state=0)
        # 建立训练集的dataloader
        train_dataset = MyDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
        # 建立测试集的dataloader
        test_dataset = MyDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle = False)
    
        #定义loss损失函数和optimizer
        model = Model(joint_dim, num_class, graph_args, edge_importance_weighting=True)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        #optimizer = torch.optim.Adam([{'params': model.parameters()}], lr = learning_rate)
        optimizer = torch.optim.SGD([{'params': model.parameters()}], lr = learning_rate)
        # Train
        for epoch in range(NUM_EPOCHS):
            train_loss = 0.0
            total_right_num = 0.0
            train_num = 0.0
            for index, epoch_data in enumerate(train_loader):
                batch_feas, batch_labels = epoch_data
                batch_feas = batch_feas.to(device)
                batch_labels = batch_labels.to(device)
                output = model(batch_feas)
            
                loss = criterion(output, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss = loss.mean().item()
                predict = torch.argmax(output, dim=1)
                assert predict.size() == batch_labels.size()
                batch_right_num = torch.sum(predict == batch_labels).item()
                train_loss += batch_loss
                total_right_num += batch_right_num
                train_num += len(predict)
                logger.info('Epoch: {:d}, Batch_index:{:d},  Loss: {:0.5f},   Acc: {:0.3f}%'.format(epoch, index, batch_loss, batch_right_num / len(predict) * 100))            
            logger.info('Epoch: {:d},   Loss: {:0.5f},   Acc: {:0.3f}%'.format(epoch, train_loss, total_right_num / train_num * 100))
        
            # Test
            logger.info('In testphase ... ')
            total_right_num = 0.0
            test_num = 0.0
            predict_labels = []
            true_labels = []
            for epoch_data in test_loader:
                batch_feas, batch_labels = epoch_data
                batch_feas = batch_feas.to(device)
                batch_labels = batch_labels.to(device)
                output = model(batch_feas)
            
                predict = torch.argmax(output, dim = 1)
                total_right_num += torch.sum(predict == batch_labels).item()
                test_num += len(predict)
                true_labels.extend(batch_labels.tolist())
                predict_labels.extend(predict.tolist())
            # use f1 
            micro_f1 = f1_score(true_labels, predict_labels, average='micro')
            macro_f1 = f1_score(true_labels, predict_labels, average='macro')
            kappa_score = cohen_kappa_score(true_labels, predict_labels)
            logger.info('Test Epoch: {:d}, micro f1: {:0.3f}, macro f1: {:0.3f}, kappa: {:0.3f}'.format(epoch, micro_f1, macro_f1, kappa_score))
            cm = confusion_matrix(true_labels, predict_labels)
            logger.info(cm)
            
            #if i == 0:
            #    total_test_acc_0.append(micro_f1)
            #    total_test_acc_1.append(macro_f1)
            if epoch == NUM_EPOCHS - 1:
                final_micro_f1_results.append(micro_f1)
                final_macro_f1_results.append(macro_f1)
                final_kappa_results.append(kappa_score)
           
        # 按照类别测试
        num_classes = 7
        logger.info('In test phase for each action...')
        true_labels_tensor = torch.tensor(true_labels)
        predict_labels_tensor = torch.tensor(predict_labels)
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
            logger.info('Class c: {:d}, micro f1: {:0.3f}, macro f1: {:0.3f}, kappa: {:0.3f}'.format(c, micro_f1, macro_f1, kappa_score))
            cm = confusion_matrix(true_labels_c, predict_labels_c)
            logger.info(cm)
