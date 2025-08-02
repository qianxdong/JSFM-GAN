import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils import utils
from PIL import Image
from tqdm import tqdm
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def adjust_map(maps):
    skin = maps[:, 1:2, :, :]
    eye_g = maps[:, 3:4, :, :]

    c = torch.chunk(eye_g, chunks=2, dim=3)
    area = torch.sum(eye_g)
    area1 = torch.sum(c[0])
    area2 = torch.sum(c[1])

    if area < 0 * (maps.shape[2] * maps.shape[3]) or area1 == 0 or area2 == 0:
        skin = skin + eye_g
        eye_g[eye_g > 0] = 0
        maps[:, 1:2, :, :] = skin
        maps[:, 3:4, :, :] = eye_g

    return maps


# python test_enhance_dir_align.py --model enhance2nd --src_dir /home/tang/work/QXD/PSFR-GAN/mydata/LR/Test_CelebA-HQ_test/ --results_dir /home/tang/work/QXD/PSFR-GAN/mydata/SR/TO/test/

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # opt.num_threads = 1  # test code only supports num_threads = 1, however just supports num_threads = 0 on cpu device
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.load_pretrain_models()

    save_dir = opt.results_dir
    os.makedirs(save_dir, exist_ok=True)

    print('creating result directory', save_dir)
    netP = model.netP
    netG = model.netG
    model.eval()
    max_size = 9999

    for i, data in tqdm(enumerate(dataset), total=len(dataset) // opt.batch_size):
        inp = data['LR']
        with torch.no_grad():
            parse_map, _ = netP(inp)
            parse_map_sm = (parse_map == parse_map.max(dim=1, keepdim=True)[0]).float().detach()
            if opt.afpm:
                parse_map_sm = adjust_map(parse_map_sm.to(opt.device))

            output_SR = netG(inp, parse_map_sm)

        img_path = data['LR_paths']  # get image paths
        for i in range(len(img_path)):
            inp_img = utils.batch_tensor_to_img(inp)
            output_sr_img = utils.batch_tensor_to_img(output_SR[0])
            ref_parse_img = utils.color_parse_map(parse_map_sm)

            save_path = os.path.join(save_dir, 'lq', os.path.basename(img_path[i]))
            os.makedirs(os.path.join(save_dir, 'lq'), exist_ok=True)
            save_img = Image.fromarray(inp_img[i])
            save_img.save(save_path)

            save_path = os.path.join(save_dir, 'hq', os.path.basename(img_path[i]))
            os.makedirs(os.path.join(save_dir, 'hq'), exist_ok=True)
            save_img = Image.fromarray(output_sr_img[i])
            save_img.save(save_path)

            save_path = os.path.join(save_dir, 'parse', os.path.basename(img_path[i]))
            os.makedirs(os.path.join(save_dir, 'parse'), exist_ok=True)
            save_img = Image.fromarray(ref_parse_img[i])
            save_img.save(save_path)

        if i > max_size:
            break
