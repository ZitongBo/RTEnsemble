import os
import zipfile
import numpy as np
np.set_printoptions(suppress=True)
import torch
from torch.utils.data.dataset import Dataset
from typing import Union, Optional
from pathlib import Path
import glob
import os
import logging
import math
import torch.nn as nn
from torch.autograd import Variable



class lstm(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, output_size=1, num_layer=2):
        super(lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)
        return x


class DianDataset(torch.utils.data.Dataset):

    def __init__(self, logger: Optional[logging.Logger], dataset_dir: Union[str, Path], dataset_type: str,
                 num_input_timestep:int, num_predict_timestep:int, means, stds) -> None:
        super().__init__()
        self.logger = logger
        assert dataset_type in ['train', 'val', 'test']
        # self.npy_data_list = np.array(sorted([os.path.join(dataset_dir, x) for x in glob.glob(dataset_dir+'/*npy')]))
        self.npy_data_list = np.array(sorted([x for x in glob.glob(dataset_dir + '/*npy')]))
        self.dataset_dir = dataset_dir
        self.dataset_type = dataset_type
        self.num_input_timestep = num_input_timestep
        self.num_predict_timestep = num_predict_timestep
        self.means = means
        self.stds = stds
        # split train/val/test
        num_npys = len(self.npy_data_list)
        # shuffle train/val/test
        npz_idxes = np.arange(num_npys)
        # np.random.shuffle(npz_idxes)
        train_split, val_split, test_split = npz_idxes[:int(num_npys*0.6)], npz_idxes[int(num_npys*0.6):int(num_npys*0.8)], \
                                                npz_idxes[int(num_npys * 0.8):]
        ##########    test  #####################################
        # train_split = npz_idxes[:int(num_npys * 0.05)]
        # val_split = npz_idxes[int(num_npys * 0.05):int(num_npys * 0.1)]
        ############################################################
        self.dataset = {'train': self.npy_data_list[train_split],
                        'val': self.npy_data_list[val_split],
                        'test': self.npy_data_list[test_split]}[self.dataset_type]

        self.buffer = [None]*len(self.dataset)
        ##############num_npys: num of npyfiles###############################
        self.num_npys = len(self.dataset)
        #############num of samples in each npy file##########################
        self.num_samples_each_npy = self.cal_total_samples(self.get_npy(0))
    def get_npy(self, np_index):
        # print(f'buffer:{len(self.buffer)} np_index:{np_index}')
        if self.buffer[np_index] is not None:
            x = self.buffer[np_index]
        else:
            npy_path = self.dataset[np_index]
            x = np.load(npy_path).transpose((0, 2, 1)).astype(np.float32)
            x[:30,:,:5] = self.means[:5]
            sb = np.zeros([x.shape[0],x.shape[1],1])
            xb = np.zeros([x.shape[0],x.shape[1],1])
            sb = x[:,:,5]*np.cos(x[:,:,6])
            xb = x[:,:,5]*np.sin(x[:,:,6])
            x[:, :, 5] = sb
            x[:, :, 6] = xb
            self.buffer[np_index] = x
        import copy
        return copy.deepcopy(x)

    def generate_train_sample(self,X, sample_idx):
        train_input = X[:, sample_idx:sample_idx+self.num_input_timestep,:]
        train_output = X[:, sample_idx+self.num_input_timestep:sample_idx+self.num_input_timestep+self.num_predict_timestep,0]
        return torch.from_numpy(train_input), torch.from_numpy(train_output)

    def cal_total_samples(self, x) -> int:
         return x.shape[1] - self.num_input_timestep-self.num_predict_timestep+1

    def __len__(self):
        return self.num_npys * self.num_samples_each_npy

    def __getitem__(self, index):
        # if self.logger:
        #     self.logger.debug(f'current sample:{index}')
        np_index, sample_index = index//self.num_samples_each_npy, index%self.num_samples_each_npy
        x = self.get_npy(np_index)
        # x = (x - self.means.reshape(1,1,-1)) / self.stds.reshape(1,1,-1)
        x[:,:,1:] = (x[:,:,1:] - self.means[1:].reshape(1, 1, -1)) / self.stds[1:].reshape(1, 1, -1)
        x = x.astype(np.float32)
        add_columns = np.arange(1, x.shape[1] + 1)
        # print(add_columns.shape)
        x = np.insert(x, 7, values=add_columns, axis=2)
        train_input, train_output = self.generate_train_sample(x, sample_index)
        sample = {
            'train_input': train_input,
            'train_output': train_output}
        # if index == 1916: print(sample)
        return sample


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def main():

if __name__=='__main__':
    import sys
    LOG_FILE = 'train.log'
    logger = logging.getLogger(__file__)
    FORMAT = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    format_str = logging.Formatter(FORMAT)  # 设置日志格式
    logger.setLevel(logging.DEBUG)  # 设置日志级别
    sh = logging.StreamHandler(stream=sys.stdout)  # 往屏幕上输出
    fh = logging.FileHandler(filename=f'LOG/{LOG_FILE}', mode='w', encoding='utf-8')
    sh.setFormatter(format_str)  # 设置屏幕上显示的格式
    fh.setFormatter(format_str)
    logger.addHandler(sh)  # 把对象加到logger里
    logger.addHandler(fh)
    num_timesteps_input = 50
    num_timesteps_output = 10
    means = np.float32([280.97805579, 122.43234318,   0.24412424,   1.38051895,   0.565941])
    stds = np.float32([1160.43493526,
    113.27333845,
    0.42102732,
    3.02794889,
    1.72929706])
    dataset = DianDataset(logger, 'data/np_dir', 'train', num_timesteps_input,
                          num_timesteps_output, means, stds)
    print(f'dataset total npys:{dataset.num_npys}, {len(dataset.buffer)}')
    # for np_indx in range(0, dataset.num_npys):
    #     print(f'{np_indx}:', dataset.get_npy(np_indx))

    model = lstm(2, 4, 1, 2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for e in range(1000):
        dataset_ = dataset.get_npy(70)
        print(dataset.dataset)
        sample = np.random.choice(a=800-10, size=128, replace=False, p=None)
        train_x,train_y = dataset.generate_train_sample(dataset.get_npy(sample), 10)
        var_x = Variable(train_x)
        var_y = Variable(train_y)
        # 前向传播
        out = model(var_x)
        loss = criterion(out, var_y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (e + 1) % 100 == 0:  # 每 100 次输出结果
            print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.data[0]))