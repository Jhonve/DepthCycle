import os
import random

import h5py
import numpy as np

from parsers import getParser
from datautils import DepthDataset

from models import create_model

from tensorboardX import SummaryWriter

k_opt = getParser()

k_loss_writer = SummaryWriter('runs/' + k_opt.name)

if not os.path.exists(k_opt.val_res_path):
    os.makedirs(k_opt.val_res_path)

if not os.path.exists(k_opt.ckpt_path):
    os.makedirs(k_opt.ckpt_path)
    os.makedirs(k_opt.ckpt_path + '/' + k_opt.name)

if not os.path.exists(k_opt.ckpt_path + '/' + k_opt.name):
    os.makedirs(k_opt.ckpt_path + '/' + k_opt.name)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = k_opt.gpu_ids

k_num_train_batch = 0
k_num_val_batch = 0

def splitData(data_path_clean, data_path_noise):
    if data_path_clean.shape[0] < data_path_noise.shape[0]:
        num_data = data_path_clean.shape[0]
    else:
        num_data = data_path_noise.shape[0]
    
    num_val_data = k_opt.num_val_batch * k_opt.batch_size
    num_train_data = num_data - num_val_data

    val_index = random.sample(range(0, num_data), num_val_data)
    train_index = list(set(range(0, num_data)) - set(val_index))

    global k_num_train_batch
    k_num_train_batch = int(len(train_index) / k_opt.batch_size)
    k_num_val_batch = int(len(val_index) / k_opt.batch_size)

    train_path_clean = data_path_clean[train_index]
    train_path_noise = data_path_noise[train_index]

    val_path_clean = data_path_clean[val_index]
    val_path_noise = data_path_noise[val_index]

    val_index = np.array(val_index)
    np.save(k_opt.val_res_path + "val_index.npy", val_index)

    return train_path_clean, train_path_noise, val_path_clean, val_path_noise    

def resplitData(data_path_clean, data_path_noise):
    if data_path_clean.shape[0] < data_path_noise.shape[0]:
        num_data = data_path_clean.shape[0]
    else:
        num_data = data_path_noise.shape[0]

    val_index = np.load(k_opt.val_res_path + "val_index.npy")
    val_index = list(val_index)
    train_index = list(set(range(0, num_data)) - set(val_index))
    
    global k_num_train_batch
    k_num_train_batch = int(len(train_index) / k_opt.batch_size)
    k_num_val_batch = int(len(val_index) / k_opt.batch_size)

    train_path_clean = data_path_clean[train_index]
    train_path_noise = data_path_noise[train_index]

    val_path_clean = data_path_clean[val_index]
    val_path_noise = data_path_noise[val_index]

    return train_path_clean, train_path_noise, val_path_clean, val_path_noise

def train():
    data_path_file_clean = k_opt.data_path_file_clean
    data_path_file_noise = k_opt.data_path_file_noise
    data_path_clean_list = h5py.File(data_path_file_clean, 'r')
    data_path_noise_list = h5py.File(data_path_file_noise, 'r')
    data_path_clean = np.array(data_path_clean_list['data_path'])
    data_path_noise = np.array(data_path_noise_list['data_path'])

    if k_opt.current_model != "" or os.path.exists(k_opt.val_res_path + "val_index.npy"):
        train_path_clean, train_path_noise, val_path_clean, val_path_noise = resplitData(data_path_clean, data_path_noise)
        print("Re-split data successfully.")
    else:
        train_path_clean, train_path_noise, val_path_clean, val_path_noise = splitData(data_path_clean, data_path_noise)
    
    # initialize Dataloader
    train_dataset = DepthDataset(k_opt, train_path_clean, train_path_noise, is_train=True)
    train_dataloader = train_dataset.getDataloader()

    # val_dataset = DepthDataset(k_opt, val_path_clean, val_path_noise, is_train=False)
    # val_dataloader = val_dataset.getDataloader()

    # initialize Network structure etc.
    current_epoch = 0
    cycleGAN_model = create_model(k_opt)
    cycleGAN_model.setup(k_opt)
    
    # cycleGAN_model.print_networks(True)
    for epoch in range(current_epoch, k_opt.num_epoch):
        for i_train, data in enumerate(train_dataloader, 0):
            cycleGAN_model.set_input(data)
            cycleGAN_model.optimize_parameters()
            loss = cycleGAN_model.get_current_losses()

            # print(i_train, loss)
            print("Epoch: %d, || Batch: %d/%d, ||" %(epoch, i_train + 1, k_num_train_batch))
            if i_train % k_opt.print_freq == 0:
                vis_imgs = cycleGAN_model.get_current_visuals()

                for loss_label, loss_value in loss.items():
                    k_loss_writer.add_scalar(loss_label, loss_value, global_step=epoch * k_num_train_batch + i_train + 1)
            
            if i_train % k_opt.display_freq == 0:
                for img_label, img_value in vis_imgs.items():
                    if img_label == 'fake_mask_rgb' or img_label == 'real_mask_rgb' or img_label == 'rec_mask_rgb':
                        vis_img = img_value.detach().cpu().numpy()
                    else:
                        vis_img = img_value[:, 0:1, :, :].detach().cpu().numpy()
                    vis_img = np.clip(vis_img, 0, 1)
                    k_loss_writer.add_images("train_"+img_label, vis_img, global_step=epoch * k_num_train_batch + i_train + 1)

        cycleGAN_model.save_networks(epoch)
        cycleGAN_model.save_networks('latest')

if __name__ == "__main__":
    train()