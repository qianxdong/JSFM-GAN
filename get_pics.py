#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：get_pics.py
@Function: get dataset from baidu.com
Created on Sun Sep 13 21:32:25 2020
@author: ydc
'''

import re
import requests
from urllib import error
from bs4 import BeautifulSoup
import os
from tqdm import tqdm
import shutil
from PIL import Image
import argparse

num = 0
numPicture = 0
file = ''
List = []


def Find(url, A):
    global List
    print('正在检测图片总数，请稍等.....')
    s = 0
    for t in tqdm(range(0, 1200, 60)):
        Url = url + str(t)
        try:
            Result = A.get(Url, timeout=7, allow_redirects=False)
        except BaseException:
            continue
        else:
            result = Result.text
            pic_url = re.findall('"objURL":"(.*?)",', result, re.S)  # 先利用正则表达式找到图片url
            s += len(pic_url)
            if len(pic_url) == 0:
                break
            else:
                List.append(pic_url)
    return s


def recommend(url):
    Re = []
    try:
        html = requests.get(url, allow_redirects=False)
    except error.HTTPError as e:
        return
    else:
        html.encoding = 'utf-8'
        bsObj = BeautifulSoup(html.text, 'html.parser')
        div = bsObj.find('div', id='topRS')
        if div is not None:
            listA = div.findAll('a')
            for i in listA:
                if i is not None:
                    Re.append(i.get_text())
        return Re


def dowmloadPicture(html, keyword):
    global num
    # t =0
    pic_url = re.findall('"objURL":"(.*?)",', html, re.S)  # 先利用正则表达式找到图片url
    # print('找到关键词:' + keyword + '的图片，即将开始下载图片...', end='')
    for each in tqdm(pic_url):
        # print('正在下载第' + str(num + 1) + '张图片，图片地址:' + str(each))
        try:
            if each is not None:
                pic = requests.get(each, timeout=7)
            else:
                continue
        except BaseException:
            print('错误，当前图片无法下载', end='')
            continue
        else:
            string = file + r'\\' + keyword + '_' + str(num) + '.jpg'
            fp = open(string, 'wb')
            fp.write(pic.content)
            fp.close()
            num += 1
        if num >= numPicture:
            return


def change_name(pics_path,name):
    pics = os.listdir(pics_path)
    i=0
    for pic in tqdm(pics):
        i = i+1
        if pic.lower().endswith(
                ('.png', '.jpg', '.jpeg')):
            # name1= pic.split('.')[0]
            origin_path = os.path.join(pics_path, pic)
            rename_path = os.path.join(pics_path, name+'_'+str(i)+'.jpg')
            if os.path.exists(rename_path):
                continue
            else:
                os.rename(origin_path, rename_path)
        else:
            raise AssertionError("The type of the pic file is wrong!")


def get_local_pic(pic_file, save_path, name,save_img):
    if not os.path.exists(pic_file):
        raise ValueError('There is no pic_file')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('Make save file successful!')
    flies = os.listdir(pic_file)
    for fl in tqdm(flies):
        # print(fl)
        if os.path.isdir(os.path.join(pic_file, fl)):
            pics = os.listdir(os.path.join(pic_file, fl))
            fname, ext = os.path.splitext(os.path.join(os.path.join(pic_file, fl), pics[0]))
            size = os.path.getsize(os.path.join(os.path.join(pic_file, fl), pics[0]))
            kb = size/1024
            if ext == '.jpg' or ext == '.png' and kb > 0:
                try:
                    if not os.path.exists(os.path.join(save_path, f'{name}_'+pics[0])):
                        shutil.copyfile(os.path.join(os.path.join(pic_file, fl), pics[0]), os.path.join(save_path, f'{name}_'+pics[0]))
                    else:
                        continue
                except:
                    pass
            else:
                pass
        else:
            size = os.path.getsize(os.path.join(pic_file, fl))
            kb = size / 1024
            if kb == 0:
                pass
            else:
                _, ext = os.path.splitext(os.path.join(pic_file, fl))
                if ext == 'jpg' or ext == 'png':
                    shutil.copyfile((os.path.join(pic_file, fl), os.path.join(save_path, f'{name}_' + fl)))
                else:
                    pass

    print('copy flies over')
    images = os.listdir(save_path)
    if not os.path.exists(save_img):
        os.makedirs(save_img)
    for im in tqdm(images):

        size = os.path.getsize(os.path.join(save_path, im))
        kb = size/1024
        if kb == 0:
            pass
        else:
            img = Image.open(os.path.join(save_path, im))
        img = img.resize((512, 512))
        img.save(os.path.join(save_img, im))
    print('resize over')



if __name__ == '__main__':  # 主函数入口
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str, default="人脸图像", help="搜索的关键词")
    parser.add_argument("--save_path", type=str, default="download", help="图像保存路径")
    args = parser.parse_args()

    # head信息
    headers = {
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0',
        'Upgrade-Insecure-Requests': '1'
    }

    A = requests.Session()
    A.headers = headers

    word = args.keyword
    url = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + word + '&pn='

    # 这里搞了下
    tot = Find(url, A)
    Recommend = recommend(url)  # 记录相关推荐
    print('经过检测%s类图片共有%d张' % (word, tot))
    numPicture = int(input('请输入想要下载的图片数量： '))
    y = os.path.exists(file)
    if y == 1:
        print('该文件已存在，请重新输入')
    else:
        file = args.save_path
        os.mkdir(file)

    t = 0
    tmp = url
    while t < numPicture:
        try:
            url = tmp + str(t)
            result = A.get(url, timeout=7, allow_redirects=False)
        except error.HTTPError as e:
            print('网络错误，请调整网络后重试')
            t = t + 60
        else:
            dowmloadPicture(result.text, word)
            t = t + 60

    print('当前搜索结束，感谢使用')
    for re in Recommend:
        print(re, end='  ')



