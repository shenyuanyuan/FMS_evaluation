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
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from random import choice
from sklearn.metrics import cohen_kappa_score
#from focal_loss import *
from logger import init_logger
log_file = r"log1\early_fusion_cnn_multi_scale_1248_dilation.txt"
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

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

#定义并训练模型
class BlockFCNConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation, momentum=0.99, epsilon=0.001, squeeze=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, padding=int(dilation*(kernel_size-1)/2), dilation = dilation)
        self.batch_norm = nn.BatchNorm1d(num_features=out_channel, eps=epsilon, momentum=momentum)
        self.relu = nn.ReLU()
    def forward(self, x):
        # input (batch_size, num_variables, time_steps), e.g. (64, 128, 545)
        x = self.conv(x)
        #import pdb;
        #pdb.set_trace();
        # input (batch_size, out_channel, L_out)
        x = self.batch_norm(x)
        # same shape as input
        y = self.relu(x)
        return y  

class BlockFCN(nn.Module):
    def __init__(self, time_steps, channels, kernels, dilation, mom=0.99, eps=0.001):
        super().__init__()
        self.conv1 = BlockFCNConv(channels[0], channels[1], kernels[0], dilation,  momentum=mom, epsilon=eps, squeeze=True)
        self.conv2 = BlockFCNConv(channels[1], channels[2], kernels[1], dilation,  momentum=mom, epsilon=eps, squeeze=True)
        output_size = time_steps - sum(kernels) + len(kernels)
        self.global_pooling = nn.AvgPool1d(kernel_size=output_size)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        # apply Global Average Pooling 1D
        y = self.global_pooling(x)
        return x, y

if __name__ == '__main__':
    front_root_path = r"E:\FMS\data\skeleton_data\front"
    side_root_path = r"E:\FMS\data\skeleton_data\side"
    score_root_path = r"E:\FMS\data\expert_score"
    sub_path = os.listdir(front_root_path)
    file_num = len(sub_path)
    front_torch_joints = []
    side_torch_joints = []
    evaluate_labels = []
    class_labels = []
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    learning_rate=0.001  #学习率
    NUM_OF_TESTS = 10
    net_params = {'channel_params':[2*joint_dim*valid_joints_num, 64, 64], 'kernel_params_1':[3, 3, 3], 'kernel_params_2':[3, 3, 3], 'kernel_params_3':[3, 3, 3], 'kernel_params_4':[3, 3, 3], 'seq_len': 545, 'num_classes': 3}
    classes_mapping = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4, 10: 5, 11: 6, 12: 6, 13: 6, 14: 6}
    json_scores = json.load(open(os.path.join(score_root_path, "experts_score.json")))
    for i in range(file_num):
        short_path = sub_path[i]
        front_file_path = os.path.join(front_root_path, short_path)
        front_load_list = sp_json(front_file_path)
        front_load_list_scale = preprocessing.scale(front_load_list)
        side_file_path = front_file_path.replace('front', 'side')
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
    front_data = rnn_utils.pad_sequence(front_torch_joints, batch_first=True, padding_value=0)
    side_data = rnn_utils.pad_sequence(side_torch_joints, batch_first=True, padding_value = 0)
    data = torch.cat((front_data, side_data), axis = 2)
    #front_dim = front_data.shape[1]
    #side_dim = side_data.shape[1]
    labels = torch.tensor(evaluate_labels, dtype=torch.int64)
    class_labels = torch.tensor(class_labels, dtype=torch.int64)
    final_micro_f1_results = []
    final_macro_f1_results = []
    final_kappa_results = []
    final_micro_f1_results_by_classes = {f'class_{index}': [] for index in range(7)}
    final_macro_f1_results_by_classes = {f'class_{index}': [] for index in range(7)}
    final_kappa_results_by_classes = {f'class_{index}': [] for index in range(7)}
    
    for i in range(NUM_OF_TESTS):
        x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(data, labels, class_labels, test_size=0.3, random_state=0)
        # 建立训练集的dataloader
        train_dataset = MyDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
        # 建立测试集的dataloader
        test_dataset = MyDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle = False)
    
        #定义loss损失函数和optimizer
        model1 = BlockFCN(net_params['seq_len'], net_params['channel_params'], net_params['kernel_params_1'], 1)
        model1 = model1.cuda()
        model2 = BlockFCN(net_params['seq_len'], net_params['channel_params'], net_params['kernel_params_2'], 2)
        model2 = model2.cuda()
        model3 = BlockFCN(net_params['seq_len'], net_params['channel_params'], net_params['kernel_params_3'], 4)
        model3 = model3.cuda()
        model4 = BlockFCN(net_params['seq_len'], net_params['channel_params'], net_params['kernel_params_4'], 8)
        model4 = model4.cuda()

        # 采用late fusion，在最后一层将两部分结果进行连接
        fc = nn.Linear(4*net_params['channel_params'][-1], net_params['num_classes'])
        fc = fc.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer =torch.optim.Adam([{'params': model1.parameters()}, {'params': model2.parameters()}, {'params': model3.parameters()}, {'params': model4.parameters()}, {'params': fc.parameters()}], lr = learning_rate)
        # Train
        for epoch in range(NUM_EPOCHS):
            train_loss = 0.0
            total_right_num = 0.0
            train_num = 0.0
            for index, epoch_data in enumerate(train_loader):
                batch_feas, batch_labels = epoch_data
                batch_feas = batch_feas.permute(0, 2, 1)
                batch_feas = batch_feas.cuda()
                batch_labels = batch_labels.cuda()
            
                fcn_conv_1, fcn_output_1 = model1(batch_feas)
                fcn_conv_2, fcn_output_2 = model2(batch_feas)
                fcn_conv_3, fcn_output_3 = model3(batch_feas)
                fcn_conv_4, fcn_output_4 = model4(batch_feas)
                #fcn_conv_5, fcn_output_5 = model5(batch_feas)

                output = fc(torch.cat((fcn_output_1.squeeze(), fcn_output_2.squeeze(), fcn_output_3.squeeze(), fcn_output_4.squeeze()), dim = 1))

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
                batch_feas = batch_feas.permute(0, 2, 1)
                batch_feas = batch_feas.cuda()
                batch_labels = batch_labels.cuda()
            
                fcn_conv_1, fcn_output_1 = model1(batch_feas)
                fcn_conv_2, fcn_output_2 = model2(batch_feas)
                fcn_conv_3, fcn_output_3 = model3(batch_feas)
                fcn_conv_4, fcn_output_4 = model4(batch_feas)
                
                output = fc(torch.cat((fcn_output_1.squeeze(), fcn_output_2.squeeze(), fcn_output_3.squeeze(), fcn_output_4.squeeze()), dim = 1))

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
            print(cm)
            logger.info(cm)
            
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
            
            logger.info('Class i: {:d}, Micro F1: {:0.3f}, macro F1: {:0.3f}, kappa: {:0.3f}'.format(c, micro_f1, macro_f1, kappa_score))
            cm = confusion_matrix(true_labels_c, predict_labels_c)
            logger.info(cm)