from PIL import Image
from PIL import ImageEnhance
import os
import cv2
import numpy as np
import math
import json
import random
import shutil
import glob


###
# 本代码共采用了四种数据增强，如采用其他数据增强方式，可以参考本代码，随意替换。
# imageDir 为原数据集的存放位置
# saveDir  为数据增强后数据的存放位置
###

def imageFlip(image_path):   #翻转图像
    img = Image.open(image_path)
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img

def rotation(image_path):
    img = Image.open(image_path)
    rotation_img = img.rotate(20) #旋转角度
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img

def randomColor(image_path): #随机颜色
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    image = Image.open(image_path)
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度


def contrastEnhancement(image_path):  # 对比度增强
    image = Image.open(image_path)
    enh_con = ImageEnhance.Contrast(image)
    # contrast = 1.5
    contrast = np.random.randint(10, 18) / 10.  # 随机因子
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted

def brightnessEnhancement(image_path):#亮度增强
    image = Image.open(image_path)
    enh_bri = ImageEnhance.Brightness(image)
    # brightness = 1.5
    brightness = np.random.randint(10, 18) / 10.  # 随机因子
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened

def colorEnhancement(image_path):#颜色增强
    image = Image.open(image_path)
    enh_col = ImageEnhance.Color(image)
    # color = 1.5
    color = np.random.randint(10, 18) / 10.  # 随机因子
    image_colored = enh_col.enhance(color)
    return image_colored

def randomCrop(image_path):
    seed = np.random.randint(0,21)
    img_size = 224
    box = (seed, seed, img_size -1 - seed, img_size - 1 - seed )

    image = Image.open(image_path)

    region = image.crop(box)
    img_croped = region.resize((img_size, img_size))

    return img_croped

def panoImagerandomCrop(image_path,cropPercent):


    image = Image.open(image_path)
    wid = image.size[0]
    hei = image.size[1]

    cw_h = int(wid*cropPercent/100)
    ch_h = int(hei*cropPercent/100)
    cw_l = int(wid * (cropPercent - 10) / 100)
    ch_l = int(hei * (cropPercent -10) / 100)
    seedw = np.random.randint(cw_l, cw_h)
    seedh = np.random.randint(ch_l, ch_h)

    box = (seedw, seedh, wid - 1 - seedw, hei - 1 - seedh)

    region = image.crop(box)
    img_croped = region.resize((wid, hei))

    filp_img = img_croped.transpose(Image.FLIP_LEFT_RIGHT)

    enh_bri = ImageEnhance.Brightness(img_croped)
    brightness = np.random.randint(10, 18) / 10.  # 随机因子
    image_brightened = enh_bri.enhance(brightness)

    enh_con = ImageEnhance.Contrast(img_croped)
    contrast = np.random.randint(10, 18) / 10.  # 随机因子
    image_contrasted = enh_con.enhance(contrast)

    return img_croped, filp_img, image_brightened, image_contrasted

def panoImageRot(image_path,rotAngle):


    img_croped = Image.open(image_path)
    # wid = image.size[0]
    # hei = image.size[1]
    #
    # cw_h = int(wid*cropPercent/100)
    # ch_h = int(hei*cropPercent/100)
    # cw_l = int(wid * (cropPercent - 10) / 100)
    # ch_l = int(hei * (cropPercent -10) / 100)
    # seedw = np.random.randint(cw_l, cw_h)
    # seedh = np.random.randint(ch_l, ch_h)
    #
    # box = (seedw, seedh, wid - 1 - seedw, hei - 1 - seedh)
    #
    # region = image.crop(box)
    # img_croped = region.resize((wid, hei))

    # rotation_img1 = img_croped.rotate(rotAngle, expand = True)  # 旋转角度
    rotation_img1 = img_croped.rotate(rotAngle)  # 旋转角度


    filp_img = img_croped.transpose(Image.FLIP_LEFT_RIGHT)

    rotation_img2 = filp_img.rotate(360-rotAngle, expand = True)  # 旋转角度
    # rotation_img2 = filp_img.rotate(360-rotAngle)  # 旋转角度


    return rotation_img1, rotation_img2


def aug_test():
    imageDir = "/home/tao/workspace/cropedFace/allcropedFaces/aug/raw"  # 要改变的图片的路径文件夹
    saveDir = "/home/tao/workspace/cropedFace/allcropedFaces/aug/out"  # 要保存的图片的路径文件夹

    for name in os.listdir(imageDir):
        saveName = name[:-4] + ".png"
        image_path = os.path.join(imageDir, name)
        image = Image.open(image_path)
        image.save(os.path.join(saveDir, saveName))

        # saveName = name[:-4] + "_cp.png"
        saveName = name[:-6] + "_0001_" + name[-5] +".png"
        saveImage = randomCrop(image_path)
        saveImage.save(os.path.join(saveDir, saveName))

        # saveName = name[:-4] + "_fl.png"
        saveName = name[:-6] + "_0002_" + name[-5] +".png"
        saveImage = imageFlip(image_path)
        saveImage.save(os.path.join(saveDir, saveName))

        # saveName = name[:-4] + "_be.png"
        saveName = name[:-6] + "_0003_" + name[-5] +".png"
        saveImage = brightnessEnhancement(image_path)
        saveImage.save(os.path.join(saveDir, saveName))

        # saveName = name[:-4] + "_ce.png"
        saveName = name[:-6] + "_0004_" + name[-5] +".png"
        saveImage = contrastEnhancement(image_path)
        saveImage.save(os.path.join(saveDir, saveName))

        # saveName = name[:-4] + "_ro.png"
        # saveImage = rotation(imageDir, name)
        # saveImage.save(os.path.join(saveDir, saveName))

def single_image_augmentation(image_path,saveDir):
    saveDir = saveDir +'/'+ image_path.split('/')[-3] + '/'+image_path.split('/')[-2]+'/'
    if not os.path.exists(os.path.dirname(saveDir)):
        os.makedirs(os.path.dirname(saveDir))
    name = image_path.split('/')[-1]
    saveName0 = name[:-6] + "_0000_" + name[-5] + ".png"
    image = Image.open(image_path)
    savePath0 = os.path.join(saveDir, saveName0)
    image.save(savePath0)

    ## 1 随机裁剪
    # saveName = name[:-4] + "_cp.png"
    saveName1 = name[:-6] + "_0001_" + name[-5] + ".png"
    saveImage = randomCrop(image_path)
    savePath1 = os.path.join(saveDir, saveName1)
    saveImage.save(savePath1)
    ## 2 左右翻转
    # saveName = name[:-4] + "_fl.png"
    saveName2 = name[:-6] + "_0002_" + name[-5] + ".png"
    saveImage = imageFlip(image_path)
    savePath2 = os.path.join(saveDir, saveName2)
    saveImage.save(savePath2)

    ## 3 调整图像亮度
    # saveName = name[:-4] + "_be.png"
    saveName3 = name[:-6] + "_0003_" + name[-5] + ".png"
    saveImage = brightnessEnhancement(image_path)
    savePath3 = os.path.join(saveDir, saveName3)
    saveImage.save(savePath3)

    ## 4 调整图像对比度
    # saveName = name[:-4] + "_ce.png"
    saveName4 = name[:-6] + "_0004_" + name[-5] + ".png"
    saveImage = contrastEnhancement(image_path)
    savePath4 = os.path.join(saveDir, saveName4)
    saveImage.save(savePath4)
    return savePath0,savePath1,savePath2,savePath3,savePath4



def shuffle_split_samplename(data_path):
    trainfakeDir = '/home/tao/workspace/cropedFace/FacesGray/grayAugTrain/fake/'
    trainrealDir = '/home/tao/workspace/cropedFace/FacesGray/grayAugTrain/real/'
    testfakeDir = '/home/tao/workspace/cropedFace/FacesGray/grayAugTest/fake/'
    testrealDir = '/home/tao/workspace/cropedFace/FacesGray/grayAugTest/real/'
    fakePath = os.path.join(data_path,'fake')
    realPath = os.path.join(data_path,'real')

    fake_name_dirs = os.listdir(fakePath)
    real_name_dirs = os.listdir(realPath)
    fake_name_count = len(fake_name_dirs)
    real_name_count = len(real_name_dirs)

    random.shuffle(fake_name_dirs)
    random.shuffle(real_name_dirs)
    train_percent = 0.7
    train_fake_num = fake_name_count*train_percent
    train_real_num = real_name_count*train_percent

    for index, namedir in enumerate(fake_name_dirs):
        if index < train_fake_num:
            shutil.move(os.path.join(fakePath,namedir)+'/',trainfakeDir)
        else:
            shutil.move(os.path.join(fakePath,namedir)+'/', testfakeDir)
    for index, namedir in enumerate(real_name_dirs):
        if index < train_real_num:
            shutil.move(os.path.join(realPath ,namedir)+'/',trainrealDir)
        else:
            shutil.move(os.path.join(realPath ,namedir)+'/',testrealDir)


def image_augmentation(image_path,inDir, saveDir):

    ## 1 随机裁剪   # ## 2 左右翻转 # ## 3 调整图像亮度 # ## 4 调整图像对比度

    saveImage1, saveImage5, saveImage9, saveImage13 = panoImagerandomCrop(image_path,10)
    savePath = image_path.replace(inDir,saveDir)
    savePath1 = savePath.replace('-ir.png','-ir-0001.png')
    saveImage1.save(savePath1)

    savePath5 = savePath.replace('-ir.png', '-ir-0005.png')
    saveImage5.save(savePath5)

    savePath9 = savePath.replace('-ir.png', '-ir-0009.png')
    saveImage9.save(savePath9)

    savePath13 = savePath.replace('-ir.png', '-ir-0013.png')
    saveImage13.save(savePath13)

    saveImage2, saveImage6, saveImage10, saveImage14 = panoImagerandomCrop(image_path, 20)
    savePath2 = savePath.replace('-ir.png', '-ir-0002.png')
    saveImage2.save(savePath2)

    savePath6 = savePath.replace('-ir.png', '-ir-0006.png')
    saveImage6.save(savePath6)

    savePath10 = savePath.replace('-ir.png', '-ir-0010.png')
    saveImage10.save(savePath10)

    savePath14 = savePath.replace('-ir.png', '-ir-0014.png')
    saveImage14.save(savePath14)

    saveImage3, saveImage7 , saveImage11, saveImage15 = panoImagerandomCrop(image_path, 30)
    savePath3 = savePath.replace('-ir.png', '-ir-0003.png')
    saveImage3.save(savePath3)

    savePath7 = savePath.replace('-ir.png', '-ir-0007.png')
    saveImage7.save(savePath7)

    savePath11 = savePath.replace('-ir.png', '-ir-0011.png')
    saveImage11.save(savePath11)

    savePath15 = savePath.replace('-ir.png', '-ir-0015.png')
    saveImage15.save(savePath15)

    saveImage4, saveImage8, saveImage12, saveImage16 = panoImagerandomCrop(image_path, 40)
    savePath4 = savePath.replace('-ir.png', '-ir-0004.png')
    saveImage4.save(savePath4)

    savePath8 = savePath.replace('-ir.png', '-ir-0008.png')
    saveImage8.save(savePath8)

    savePath12 = savePath.replace('-ir.png', '-ir-0012.png')
    saveImage12.save(savePath12)

    savePath16 = savePath.replace('-ir.png', '-ir-0016.png')
    saveImage16.save(savePath16)

    # ## 3 调整图像亮度
    saveImage17 = brightnessEnhancement(image_path)
    savePath17 = savePath.replace('-ir.png', '-ir-0017.png')
    saveImage17.save(savePath17)

    # ## 4 调整图像对比度
    saveImage18 = contrastEnhancement(image_path)
    savePath18 = savePath.replace('-ir.png', '-ir-0018.png')
    saveImage18.save(savePath18)


def _image_augmentation(image_path,inDir, saveDir):

    ## 1 随机裁剪   # ## 2 左右翻转 # ## 3 调整图像亮度 # ## 4 调整图像对比度

    saveImage1, saveImage5= panoImageRot(image_path,10)
    savePath = image_path.replace(inDir,saveDir)
    savePath1 = savePath.replace('_b.png','_b_0019.png')
    saveImage1.save(savePath1)

    savePath5 = savePath.replace('_b.png', '_b_0020.png')
    saveImage5.save(savePath5)


    saveImage2, saveImage6 = panoImageRot(image_path, 20)
    savePath2 = savePath.replace('_b.png', '_b_0021.png')
    saveImage2.save(savePath2)

    savePath6 = savePath.replace('_b.png', '_b_0022.png')
    saveImage6.save(savePath6)


    saveImage3, saveImage7 = panoImageRot(image_path, 30)
    savePath3 = savePath.replace('_b.png', '_b_0023.png')
    saveImage3.save(savePath3)

    savePath7 = savePath.replace('_b.png', '_b_0024.png')
    saveImage7.save(savePath7)


    saveImage4, saveImage8 = panoImageRot(image_path, 40)
    savePath4 = savePath.replace('_b.png', '_b_0025.png')
    saveImage4.save(savePath4)

    savePath8 = savePath.replace('_b.png', '_b_0026.png')
    saveImage8.save(savePath8)


def bctc_datas_augmentation(input_dir, save_dir):
    input_path = input_dir +'/*.png'
    imgs = sorted(glob.glob(input_path))
    print(len(imgs))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for img_path in imgs:
        _image_augmentation(img_path,input_dir, save_dir)


if __name__=='__main__':

    input_dir = '/home/tao/workspace/YZhang/3DFace/bctcDebug/images0901/3Dtoumoir/fake20'
    save_dir = input_dir + '_rot'
    bctc_datas_augmentation(input_dir,save_dir)




