import os

import h5py
import numpy as np

from parsers import getParser
from datautils import DepthDataset
from utils import saveImg

from models import create_model

k_opt = getParser()

if not os.path.exists(k_opt.test_res_path):
    os.makedirs(k_opt.test_res_path)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = k_opt.gpu_ids

def test():
    data_path_file_clean = k_opt.data_path_file_clean
    data_path_file_noise = k_opt.data_path_file_noise
    data_path_clean = h5py.File(data_path_file_clean, 'r')
    data_path_noise = h5py.File(data_path_file_noise, 'r')
    data_path_clean = np.array(data_path_clean['data_path'])
    data_path_noise = np.array(data_path_noise['data_path'])

    num_test_batch = int(len(data_path_clean) / k_opt.batch_size)

    test_dataset = DepthDataset(k_opt, data_path_clean, data_path_clean, is_train=False)
    test_dataloader = test_dataset.getDataloader()

    cycleGAN_model = create_model(k_opt)
    cycleGAN_model.setup(k_opt)

    if k_opt.eval:
        cycleGAN_model.eval()

    for i_test, data in enumerate(test_dataloader, 0):
        cycleGAN_model.set_input(data)
        cycleGAN_model.test()

        vis_imgs = cycleGAN_model.get_current_visuals()

        vis_imgs = vis_imgs['fake'].permute(0, 2, 3, 1).detach().cpu().numpy()
        paths = cycleGAN_model.get_image_paths()

        for i_batch in range(k_opt.batch_size):
            img = vis_imgs[i_batch]
            path = paths['A_paths'][i_batch].split('|')[0]
            path = path.split('/')
            scene_name = path[5]
            img_name = path[8]

            if not os.path.exists(k_opt.test_res_path + '/' + scene_name):
                os.makedirs(k_opt.test_res_path + '/' + scene_name)

            save_path = k_opt.test_res_path + '/' + scene_name + '/' + img_name
            saveImg(save_path, img)

        print(i_test, "Saving Done %d/%d" %(i_test + 1, num_test_batch))

if __name__ == "__main__":
    test()