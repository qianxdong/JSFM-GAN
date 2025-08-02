#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：test_enhance_video.py
@Author  ：Xiaodong Qian
@Function:
@Date    ：2023/7/18 19:36 
'''

from options.test_options import TestOptions
from models import create_model
import torch
import cv2
import dlib
from test_enhance_single_unalign import detect_video_align_faces, video_faces_back
from utils import utils

def preact(LQ_faces, model):
    hq_faces = []
    for lq_face in LQ_faces:
        with torch.no_grad():
            lq_tensor = torch.tensor(lq_face.transpose(2, 0, 1)) / 255. * 2 - 1
            lq_tensor = lq_tensor.unsqueeze(0).float().to(model.device)
            parse_map, _ = model.netP(lq_tensor)
            parse_map_onehot = (parse_map == parse_map.max(dim=1, keepdim=True)[0]).float()
            output_SR = model.netG(lq_tensor, parse_map_onehot)
        hq_faces.append(utils.tensor_to_img(output_SR[0]))
    return hq_faces


if __name__ == '__main__':
    # 检测、截取、对齐
    if torch.cuda.is_available():
        face_detector = dlib.cnn_face_detection_model_v1('./pretrain_models/mmod_human_face_detector.dat')
    else:
        face_detector = dlib.get_frontal_face_detector()  # cpu
    lmk_predictor = dlib.shape_predictor('./pretrain_models/shape_predictor_5_face_landmarks.dat')
    template_path = './pretrain_models/FFHQ_template.npy'

    # enhance 模型
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0  # test code only supports num_threads = 1, however just supports num_threads = 0 on cpu device
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True

    model = create_model(opt)  # create a model given opt.model and other options
    model.load_pretrain_models()
    print('加载权重成功！')

    model.eval()
    max_size = 9999

    # 视频设置
    video_path = opt.src_dir
    video_save_path = opt.results_dir

    capture = cv2.VideoCapture(video_path)
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取原视频的宽
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取原视频的高
    # fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))  # 视频的编码
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # out = cv2.VideoWriter(video_save_path, fourcc, video_fps, (575-245, 470-50))
    out = cv2.VideoWriter(video_save_path, fourcc, video_fps, (width, height))

    print('开始恢复视频！')
    count = 0
    while True:
        count += 1
        print(f"{count}/{frames}帧")
        ret, frame = capture.read()
        # if count == 2:
        #     break

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            # frame = cv2.resize(frame, (width//2, height//2))
            # frame = frame[50:470, 245:575, :]
            aligned_faces, tform_params = detect_video_align_faces(frame, face_detector, lmk_predictor, template_path)

            hq_faces = preact(aligned_faces, model)
            frame = video_faces_back(frame, hq_faces, tform_params, upscale=1)

            # frame[80:400, 270:550, :] = face_region
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # frame = cv2.resize(frame, (width, height))
            out.write(frame)
        else:
            break
    capture.release()
    out.release()
    print(f'恢复结束！请在{video_save_path}查看结果')
