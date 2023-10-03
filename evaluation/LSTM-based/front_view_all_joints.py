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
from logger import init_logger
from sklearn.metrics import cohen_kappa_score
log_file = r"log1\front_view_lstm_atten_log.txt"
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
class RNN(nn.Module):
    def __init__(self, list_params):
        super(RNN,self).__init__()
        self.rnn = nn.LSTM(
          input_size = list_params['channel_params'][0],
          hidden_size = list_params['channel_params'][1],
          num_layers = 2, batch_first=True, dropout=0.5, bidirectional=True)
        self.out = nn.Linear(2*list_params['channel_params'][1], list_params['num_classes'])
        self.w_omega_1 = nn.Parameter(torch.Tensor(list_params['channel_params'][1], list_params['channel_params'][1]))
        self.w_omega_2 = nn.Parameter(torch.Tensor(list_params['channel_params'][1], list_params['channel_params'][1]))
        self.u_omega_1 = nn.Parameter(torch.Tensor(list_params['channel_params'][1], 1))
        self.u_omega_2 = nn.Parameter(torch.Tensor(list_params['channel_params'][1], 1))
        nn.init.uniform_(self.w_omega_1, -0.1, 0.1)
        nn.init.uniform_(self.u_omega_1, -0.1, 0.1)
        nn.init.uniform_(self.w_omega_2, -0.1, 0.1)
        nn.init.uniform_(self.u_omega_2, -0.1, 0.1)

    def attention_net(self, x, w_omega, u_omega):
        u = torch.tanh(torch.matmul(x, w_omega))
        att = torch.matmul(u, u_omega)
        att_score = F.softmax(att, dim = 1)
        scored_x = x*att_score
        output = torch.sum(scored_x, dim = 1)
        return output

    def forward(self,x):
        r_out, h_n = self.rnn(x)
        r_out = r_out.view(r_out.shape[0], r_out.shape[1], 2, -1)
        forward_weight_r_out = self.attention_net(r_out[:, :, 0, :], self.w_omega_1, self.u_omega_1)
        reverse_weight_r_out = self.attention_net(r_out[:, :, 1, :], self.w_omega_2, self.u_omega_2)
        weight_r_out = torch.concat((forward_weight_r_out, reverse_weight_r_out), dim = -1)
        output = self.out(weight_r_out.reshape(len(weight_r_out), -1))
        return output

def adjust_learning_rate(optimizer, epoch, start_lr):
    lr = start_lr * (0.1 ** (epoch//20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
    NUM_EPOCHS = 100
    NUM_OF_TESTS = 10
    net_params = {'channel_params':[joint_dim*valid_joints_num, 32, 64], 'seq_len': 545, 'num_classes': 3}
    classes_mapping = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4, 10: 5, 11: 6, 12: 6, 13: 6, 14: 6}
    json_scores = json.load(open(os.path.join(score_root_path, "experts_score.json")))
    for i in range(file_num):
        short_path = sub_path[i]
        front_file_path = os.path.join(front_root_path, short_path)
        front_load_list = sp_json(front_file_path)
        front_load_list_scale = preprocessing.scale(front_load_list)
        #side_file_path = front_file_path.replace('front', 'side')
        #side_load_list = sp_json(side_file_path)
        #side_load_list_scale = preprocessing.scale(side_load_list)
        
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
            #side_torch_joints.append(torch.tensor(side_load_list_scale, dtype=torch.float))
            evaluate_labels.append(final_score - 1)
            class_labels.append(classes_mapping[int(action_index[1:])-1]) 
    data = rnn_utils.pad_sequence(front_torch_joints, batch_first=True, padding_value=0)
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
        model = RNN(net_params)
        model.cuda()
        criterion = nn.CrossEntropyLoss()
        start_lr = 0.005  #学习率
        learning_rate = 0.001
        optimizer = torch.optim.Adam([{'params': model.parameters()}], lr = learning_rate)
        # Train
        for epoch in range(NUM_EPOCHS):
            #adjust_learning_rate(optimizer, epoch, start_lr)
            #print("Epoch: {} lr: {: .2E}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
            train_loss = 0.0
            total_right_num = 0.0
            train_num = 0.0
            for index, epoch_data in enumerate(train_loader):
                batch_feas, batch_labels = epoch_data
                batch_feas = batch_feas.cuda()
                batch_labels = batch_labels.cuda()
                
                #fcn_conv, fcn_output = model(batch_feas)
                output = model(batch_feas)
                
                loss = criterion(output, batch_labels)
                loss.cuda()
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
                batch_feas = batch_feas.cuda()
                batch_labels = batch_labels.cuda()
                #batch_feas = batch_feas.permute(0, 2, 1)
            
                output = model(batch_feas)
                #output = fc(fcn_output.squeeze())
            
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