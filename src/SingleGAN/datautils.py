import os
import random

import numpy as np
import scipy.io as sio
import h5py
import cv2

import torch
import torch.utils.data as data

from parsers import getParser

k_data_path_clean = '../../Dataset/InteriorNet/random/'
k_data_path_noise = '../../Dataset/ScanNet/'
k_h5_path = './'

k_opt = getParser()

class DepthDataset(data.Dataset):
    def __init__(self, data_opt, data_path_clean, data_path_noise, is_train):
        super(DepthDataset).__init__()
        self.batch_size = data_opt.batch_size
        self.num_workers = data_opt.num_workers
        self.data_path_clean = data_path_clean
        self.data_path_noise = data_path_noise

        if len(self.data_path_clean) < len(self.data_path_noise):
            self.SIZE = len(self.data_path_clean)
        else:
            self.SIZE = len(self.data_path_noise)

        self.is_train = is_train

    def __len__(self):
        return self.SIZE

    def loadRGBDImg(self, data_path):
        data_path_depth = data_path.split('|')[0]
        data_path_rgb = data_path.split('|')[1]

        depth_img = cv2.imread(data_path_depth, 2)
        depth_img = cv2.resize(depth_img, (int(depth_img.shape[1] / k_opt.zoom_out_scale), int(depth_img.shape[0] / k_opt.zoom_out_scale)), interpolation=cv2.INTER_AREA)
        depth_img = np.array(depth_img).astype(np.float32)
        depth_max = np.max(depth_img)

        depth_img = depth_img / depth_max

        rgb_img = cv2.imread(data_path_rgb)
        rgb_img = cv2.resize(rgb_img, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_AREA)
        rgb_img = rgb_img / 255.
        rgb_img = rgb_img.astype(np.float32)

        depth_img = np.expand_dims(depth_img, 2)

        rgbd_img = np.concatenate((depth_img, rgb_img), 2)

        rgbd_img = rgbd_img.transpose(2, 0, 1)
        return rgbd_img

    def __getitem__(self, index):
        data_item = {}
        data_item['A'] = self.loadRGBDImg(self.data_path_clean[index])
        data_item['B'] = self.loadRGBDImg(self.data_path_noise[index])
        data_item['A_paths'] = self.data_path_clean[index]
        data_item['B_paths'] = self.data_path_noise[index]
        return data_item

    def getDataloader(self):
        if(self.is_train):
            return data.DataLoader(dataset=self, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)
        else:
            return data.DataLoader(dataset=self, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, drop_last=True)

def preDataPathInterior(data_path, folder_list):
    all_file_list = []
    for i in range(len(folder_list)):
        file_list_depth = os.listdir(data_path + folder_list[i] + '/depth0/data')
        for j in range(len(file_list_depth)):
            file_rgb = data_path + folder_list[i] + '/cam0/data/' + file_list_depth[j]
            file_list_depth[j] = data_path + folder_list[i] + '/depth0/data/' + file_list_depth[j]
            paired_file_name = file_list_depth[j] + '|' + file_rgb
            all_file_list.append(paired_file_name)
    
    return all_file_list

def preDataPathScanNet(data_path, folder_list):
    all_file_list = []
    for i in range(len(folder_list)):
        file_list_depth = os.listdir(data_path + folder_list[i] + '/depth')
        for j in range(len(file_list_depth)):
            file_rgb = data_path + folder_list[i] + '/color/' + file_list_depth[j].split('.')[0] + '.jpg'
            file_list_depth[j] = data_path + folder_list[i] + '/depth/' + file_list_depth[j]
            paired_file_name = file_list_depth[j] + '|' + file_rgb
            all_file_list.append(paired_file_name)
    
    return all_file_list

def saveH5(files_path, target_name='dataPath.h5'):
    files_path = np.array(files_path)

    with h5py.File(k_h5_path + target_name, 'w') as data_file:
        data_type = h5py.special_dtype(vlen=str)
        data = data_file.create_dataset('data_path', files_path.shape, dtype=data_type)
        data[:] = files_path
        data_file.close()

    print('Save path done!')

def datapathPrepare():
    # initialize data path to dataPath.h5
    folder_list_clean = os.listdir(k_data_path_clean)
    print('Number of clean models: ', len(folder_list_clean))

    files_path_clean = preDataPathInterior(k_data_path_clean, folder_list_clean)
    print('Number o clean data: ', len(files_path_clean))
    saveH5(files_path_clean, 'DataPathClean.h5')
    
    '''
    folder_list_noise = os.listdir(k_data_path_noise)
    print('Number of noise folders: ', len(folder_list_noise))

    files_path_noise = preDataPathScanNet(k_data_path_noise, folder_list_noise)
    print('Number of noise data: ', len(files_path_noise))
    saveH5(files_path_noise, 'DataPathNoise.h5')
    '''

def selfTest():
    data_path_file_clean = k_opt.data_path_file_clean
    data_path_file_noise = k_opt.data_path_file_noise

    data_path_clean = h5py.File(data_path_file_clean, 'r')
    data_path_noise = h5py.File(data_path_file_noise, 'r')

    data_path_clean = np.array(data_path_clean['data_path'])
    data_path_noise = np.array(data_path_noise['data_path'])

    dataset = DepthDataset(k_opt, data_path_clean, data_path_noise, False)

    data_size = int(dataset.__len__() / k_opt.batch_size)

    data_loader = dataset.getDataloader()

    for i, data in enumerate(data_loader, 0):
        A = data['A']
        B = data['B']
        print(i+1, '/',  data_size, A.shape, B.shape)
        if A.shape[2] != 240 or B.shape[2] != 240:
            break

    
if __name__ == '__main__':
    # datapathPrepare()
    selfTest()