import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils import utils
from PIL import Image
from tqdm import tqdm
import torch
from torch.backends import cudnn
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# python test_enhance_dir_align_eval.py
if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0  # test code only supports num_threads = 1, however just supports num_threads = 0 on cpu device
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True

    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    # model.load_pretrain_models()

    # save_dir = opt.results_dir
    # os.makedirs(save_dir, exist_ok=True)

    # print('creating result directory', save_dir)
    cudnn.benchmark = True
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1000
    timings = np.zeros((repetitions, 1))
    device = torch.device('cuda:0')
    netP = model.netP.to(device)
    netG = model.netG.to(device)
    model.eval()
    # print(netG)

    dummy_input = torch.randn(1, 3, 512, 512, dtype=torch.float).to(device)
    mask_input = torch.randn(1, 19, 512, 512, dtype=torch.float).to(device)
    # gpu 预热
    with torch.no_grad():
        for _ in range(10):
            _ = netP(dummy_input)
    # 测试
    # with torch.no_grad():
    #     for rep in range(repetitions):
    #         starter.record()
    #         _ = netP(dummy_input)
    #         ender.record()
    #         torch.cuda.synchronize()
    #         curr_time = starter.elapsed_time(ender)
    #         timings[rep] = curr_time

    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = netG(dummy_input, mask_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    avg = timings.sum() / repetitions
    print('\navg={}\n'.format(avg))


