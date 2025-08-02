#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：v.py
@Author  ：Xiaodong Qian
@Function:
@Date    ：2023/7/19 10:06 
'''
import os
import cv2
import numpy as np


def multi_video_to_single(video1_path, video2_path, save_path):
    # 判断路径是否存在
    if not os.path.exists(video1_path) or not os.path.exists(video1_path):
        raise ValueError("视频路径不存在，请检查视频路径。")

    save_video_path = save_path

    # 读取视频
    capture1 = cv2.VideoCapture(video1_path)
    capture2 = cv2.VideoCapture(video2_path)

    # 获取视频属性，帧数、帧率、宽高
    video1_fps = capture1.get(cv2.CAP_PROP_FPS)
    # frames1 = capture1.get(cv2.CAP_PROP_FRAME_COUNT)
    width1 = int(capture1.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取原视频的宽
    height1 = int(capture1.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取原视频的高

    video2_fps = capture2.get(cv2.CAP_PROP_FPS)
    # frames2 = capture2.get(cv2.CAP_PROP_FRAME_COUNT)
    width2 = int(capture2.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取原视频的宽
    height2 = int(capture2.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取原视频的高

    # 检查属性
    # if video1_fps != video2_fps or frames1 != frames2 or width1 != width2 or height1 != height2:
    #     raise ValueError("视频属性不合符要求。")
    if height1 != height2:
        raise ValueError(f"视频属性不合符要求。\n"
                         f"帧率：{video1_fps}：{video2_fps}; \n"
                         f"帧大小：({width1},{height1})：({width2},{height2})")

    # 保存的编码、写入
    front_size = 1
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_video_path, fourcc, video1_fps, (width2+width1, height1))

    while True:
        ret1, frame1 = capture1.read()
        ret2, frame2 = capture2.read()

        # 两个视频都读到
        if ret1 and ret2:
            frame = np.hstack((frame1, frame2))
            cv2.line(frame, (width1, 0), (width1, height1), (255, 255, 255), front_size)
            out.write(frame)
        else:
            break
    capture1.release()
    capture2.release()
    out.release()


def contrast_video(video1_path, video2_path, save_path):
    # 判断路径是否存在
    if not os.path.exists(video1_path) or not os.path.exists(video1_path):
        raise ValueError("视频路径不存在，请检查视频路径。")
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # save_video_path = os.path.join(save_path, "contract.mp4")
    save_video_path = save_path

    # 读取视频
    capture1 = cv2.VideoCapture(video1_path)
    capture2 = cv2.VideoCapture(video2_path)

    # 获取视频属性，帧数、帧率、宽高
    video1_fps = capture1.get(cv2.CAP_PROP_FPS)
    # frames1 = capture1.get(cv2.CAP_PROP_FRAME_COUNT)
    width1 = int(capture1.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取原视频的宽
    height1 = int(capture1.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取原视频的高

    video2_fps = capture2.get(cv2.CAP_PROP_FPS)
    # frames2 = capture2.get(cv2.CAP_PROP_FRAME_COUNT)
    width2 = int(capture2.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取原视频的宽
    height2 = int(capture2.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取原视频的高

    # 检查属性
    # if video1_fps != video2_fps or frames1 != frames2 or width1 != width2 or height1 != height2:
    #     raise ValueError("视频属性不合符要求。")
    if width1 != width2 or height1 != height2:
        raise ValueError(f"视频属性不合符要求。\n"
                         f"帧率：{video1_fps}：{video2_fps}; \n"
                         f"帧大小：({width1},{height1})：({width2},{height2})")


    # 保存的编码、写入
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_video_path, fourcc, video1_fps, (width1, height1))
    front_size = 1

    # 循坏写入视频
    start = 0
    left_flag = True
    while True:
        # 统计对比的起始坐标
        if start < width1 and left_flag:
            if width1 // 3 < start < width1//3 * 2:
                start += 2
            else:
                start += 3
        else:
            left_flag = False
            if width1 // 3 < start < width1 // 3 * 2:
                start -= 2
            else:
                start -= 3
            if start < 0:
                left_flag = True

        ret1, frame1 = capture1.read()
        ret2, frame2 = capture2.read()

        # 两个视频都读到
        if ret1 and ret2:
            cv2.line(frame1, (start, 0), (start, height1), (255, 255, 255), front_size)
            frame1[:, start + front_size:, :] = frame2[:, start + front_size:, :]
            out.write(frame1)
        else:
            break
    capture1.release()
    capture2.release()
    out.release()


def get_audio_from_video(video_path, save_path):
    from moviepy.editor import VideoFileClip
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(save_path)


def add_audio_into_video(video_path, audio_path,save_path):
    from moviepy.editor import AudioFileClip, VideoFileClip
    ad = AudioFileClip(audio_path)
    vd = VideoFileClip(video_path)
    vd2 = vd.set_audio(ad)  # 将提取到的音频和视频文件进行合成
    vd2.write_videofile(save_path)  # 输出新的视频文件
    vd2.write_videofile(save_path, audio_bitrate="1000k", bitrate="2000k")

def video2gif(video_path, save_path):
    from moviepy.editor import VideoFileClip
    clip = VideoFileClip(video_path).subclip(0, 5).resize(0.6)  # 宽度和高度乘以 0.1   # 1~3s
    clip.write_gif(save_path)

def convert_mp4_to_gif(video_path,save_path,duration=80,step=3):
    from PIL import Image
    video_capture = cv2.VideoCapture(video_path)
    i = 0
    frames = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        if i % step == 0:
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame)
        i += 1
    frame_one = frames[0]
    frame_one.save(save_path,
                   format="GIF",
                   append_images=frames[1:],
                   save_all=True,
                   duration=duration,
                   loop=0,
                   disposal=0)
    # loop=0  GIF循环的整数次数。0 表示它将循环永远
    # disposal=2 恢复原背景颜色。0 - 未指定处置。1 - 不要丢弃。2 - 恢复到背景颜色。3 - 恢复到以前的内容。
    # save_all  true:保存图像的所有帧。false:保存多帧图像的第一帧。;
    # transparency 透明度颜色指数;
    # optimize true，压缩调色板消除未使用的颜色
    # duration 单位毫秒，每帧的显示持续时间。传递一个常量持续时间的单个整数，或 列表或元组以分别设置每个帧的持续时间。
    # append_images 要追加为附加帧的图像列表。每个列表中的图像可以是单帧或多帧图像
    # palette 对保存的图像使用指定的调色板
    # https://pillow.readthedocs.io/en/latest/handbook/image-file-formats.html#gif




if __name__ == '__main__':
    # # 获取音频
    # video_path = r'D:\w\GCFSR_MY\video/head.mp4'
    mp3_path = r'D:\w\GCFSR_MY\video/videos/head.mp3'
    # get_audio_from_video(video_path, mp3_path)

    # # 增强对比合成视频
    pre_video_path = r'D:\w\GCFSR_MY\video/videos/head.mp4'
    enhance_video_path = r'D:\w\GCFSR_MY\video/videos/head_enhance.mp4'
    compare_video_path = r'D:\w\GCFSR_MY\video/videos/head_contrast.mp4'
    single_video_path = r'D:\w\GCFSR_MY\video/3head_contrast.mp4'
    dsingle_video_path = r'D:\w\GCFSR_MY\video/4head_contrast.mp4'
    # contrast_video(pre_video_path, enhance_video_path, compare_video_path)
    # multi_video_to_single(single_video_path, compare_video_path, dsingle_video_path)
    #
    # # 音频融合到合成视频
    result_video_path = r'D:\w\GCFSR_MY\video/result.mp4'
    path = r'D:\w\GCFSR_MY\video/contrast.mp4'
    add_audio_into_video(dsingle_video_path, mp3_path, path)

    # video转gif
    # gifpath = r'D:\w\GCFSR_MY\video/enhence.gif'
    # video2gif(compare_video_path, gifpath)
    # convert_mp4_to_gif(compare_video_path, gifpath)