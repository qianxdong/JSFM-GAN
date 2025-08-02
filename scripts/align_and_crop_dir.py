import dlib
import os
import cv2
import numpy as np 
from tqdm import tqdm
from skimage import transform as trans
from skimage import io
import argparse


def get_points(img, detector, shape_predictor, size_threshold=999):
    dets = detector(img, 1)
    if len(dets) == 0:
        return None
    
    all_points = []
    for det in dets:
        if isinstance(detector, dlib.cnn_face_detection_model_v1):
            rec = det.rect # for cnn detector
        else:
            rec = det
        if rec.width() > size_threshold or rec.height() > size_threshold: 
            break
        shape = shape_predictor(img, rec) 
        single_points = []
        for i in range(5):
            single_points.append([shape.part(i).x, shape.part(i).y])
        all_points.append(np.array(single_points))
    if len(all_points) <= 0:
        return None
    else:
        return all_points

def align_and_save(img, save_path, src_points, template_path, template_scale=1):
    out_size = (512, 512)
    reference = np.load(template_path) / template_scale

    ext = os.path.splitext(save_path)
    for idx, spoint in enumerate(src_points):
        tform = trans.SimilarityTransform()
        tform.estimate(spoint, reference)
        M = tform.params[0:2, :]

        crop_img = cv2.warpAffine(img, M, out_size)
        if len(src_points) > 1:
            save_path = ext[0] + '_{}'.format(idx) + ext[1]
        dlib.save_image(crop_img.astype(np.uint8), save_path)
        # print('Saving image', save_path)

def align_and_save_dir(src_dir, save_dir, template_path='./pretrain_models/FFHQ_template.npy', template_scale=2, use_cnn_detector=True):
    from tqdm import tqdm
    # from image_degradation import process_image
    # out_size = (512, 512)

    if use_cnn_detector:
        detector = dlib.cnn_face_detection_model_v1('./pretrain_models/mmod_human_face_detector.dat')
    else:
        detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor('./pretrain_models/shape_predictor_5_face_landmarks.dat')

    for name in tqdm(os.listdir(src_dir)):
        img_path = os.path.join(src_dir, name)

        img = cv2.imread(img_path)  # 避免图片太小以及图片读取无内容
        if img is None:
            continue
        else:
            size = img.shape
            if max(size[0], size[1]) < 20:
                continue
            # if max(size[0], size[1]) > 2000:
            #     img = process_image(img, 1024)
            #     cv2.imwrite(img_path, img)


        imgg = dlib.load_rgb_image(img_path)

        points = get_points(imgg, detector, sp)
        if points is not None:
            save_path = os.path.join(save_dir, name)
            align_and_save(imgg, save_path, points, template_path, template_scale)
        else:
            no_face = 'mydata/No_faces'
            if not os.path.exists(no_face):
                os.makedirs(no_face)
            no_face_path = os.path.join(no_face, name)
            cv2.imwrite(no_face_path, img)
            # print('No face detected in', img_path)
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, default=r'C:\Users\Administrator\Desktop\motion', help='source directory containing images to crop and align.')
    parser.add_argument('--results_dir', type=str, default=r'C:\Users\Administrator\Desktop\motion_aling', help='results directory to save the aligned faces.')
    parser.add_argument('--use_cnn_detector', action='store_true', help='do not use cnn face detector in dlib.')
    args = parser.parse_args()

    src_dir = args.src_dir
    assert os.path.isdir(src_dir), 'Source path should be a directory containing images'
    save_dir = args.results_dir
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
    align_and_save_dir(src_dir, save_dir)



            

