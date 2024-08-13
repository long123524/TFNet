import os
import random
import shutil


def moveFile(input1, input2, input3,input4,save1, save2,save3,save4):
    pathDir = os.listdir(input1)  # 取图片的原始路径
    random.seed(1)
    filenumber = len(pathDir)  # 原文件个数
    rate = 0.2# 抽取的验证集的比例，占总数据的多少
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取需要数量的样本图片
    print(sample)
    list_len = len(sample)
    print(list_len)
    list = []
    for i in range(len(sample)):
        list.append(sample[i].split('.')[0])
    print(list)
    for flie_name in list:
        path_img = os.path.join(input1, flie_name + '.tif')
        shutil.move(path_img, save1)
        path_lab = os.path.join(input2, flie_name + '.tif')
        shutil.move(path_lab, save2)
        path_lab = os.path.join(input3, flie_name + '.tif')
        shutil.move(path_lab, save3)
        path_lab = os.path.join(input4, flie_name + '.tif')
        shutil.move(path_lab, save4)

def moveFile2(input1, input2, input3,input4,input5,save1, save2,save3,save4,save5):
    pathDir = os.listdir(input1)  # 取图片的原始路径
    random.seed(1)
    filenumber = len(pathDir)  # 原文件个数
    rate = 0.2# 抽取的验证集的比例，占总数据的多少
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取需要数量的样本图片
    print(sample)
    list_len = len(sample)
    print(list_len)
    list = []
    for i in range(len(sample)):
        list.append(sample[i].split('.')[0])
    print(list)
    for flie_name in list:
        path_img = os.path.join(input1, flie_name + '.tif')
        shutil.move(path_img, save1)
        path_lab = os.path.join(input2, flie_name + '.tif')
        shutil.move(path_lab, save2)
        path_lab = os.path.join(input3, flie_name + '.tif')
        shutil.move(path_lab, save3)
        path_lab = os.path.join(input4, flie_name + '.tif')
        shutil.move(path_lab, save4)
        path_lab = os.path.join(input5, flie_name + '.tif')
        shutil.move(path_lab, save5)

def moveFile_4(input1, input2, input3,input4,input5,input6,save1, save2,save3,save4,save5,save6):
    pathDir = os.listdir(input1)  # 取图片的原始路径
    random.seed(1)
    filenumber = len(pathDir)  # 原文件个数
    rate = 0.2# 抽取的验证集的比例，占总数据的多少
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取需要数量的样本图片
    print(sample)
    list_len = len(sample)
    print(list_len)
    list = []
    for i in range(len(sample)):
        list.append(sample[i].split('.')[0])
    print(list)
    for flie_name in list:
        path_img = os.path.join(input1, flie_name + '.tif')
        shutil.move(path_img, save1)
        path_img = os.path.join(input2, flie_name + '.tif')
        shutil.move(path_img, save2)
        path_lab = os.path.join(input3, flie_name + '.tif')
        shutil.move(path_lab, save3)
        path_lab = os.path.join(input4, flie_name + '.tif')
        shutil.move(path_lab, save4)
        path_lab = os.path.join(input5, flie_name + '.tif')
        shutil.move(path_lab, save5)
        path_lab = os.path.join(input6, flie_name + '.tif')
        shutil.move(path_lab, save6)


def moveFile_3(input1, input2, input3,save1, save2,save3):
    pathDir = os.listdir(input1)  # 取图片的原始路径
    random.seed(1)
    filenumber = len(pathDir)  # 原文件个数
    rate = 0.2# 抽取的验证集的比例，占总数据的多少
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取需要数量的样本图片
    print(sample)
    list_len = len(sample)
    print(list_len)
    list = []
    for i in range(len(sample)):
        list.append(sample[i].split('.')[0])
    print(list)
    for flie_name in list:
        path_img = os.path.join(input1, flie_name + '.tif')
        shutil.move(path_img, save1)
        path_lab = os.path.join(input2, flie_name + '.tif')
        shutil.move(path_lab, save2)
        path_lab = os.path.join(input3, flie_name + '.tif')
        shutil.move(path_lab, save3)


if __name__ == '__main__':
    input_path1 = r'D:\longJ\SD\train\image'
    input_path2 = r'D:\longJ\SD\train\mask'
    input_path3 = r'D:\longJ\SD\train\contour'

    save_img1 = r'D:\longJ\SD\valid\image'
    save_img2 = r'D:\longJ\SD\valid\mask'
    save_lab1 = r'D:\longJ\SD\valid\contour'

    moveFile_3(input_path1, input_path2,input_path3, save_img1, save_img2,save_lab1)