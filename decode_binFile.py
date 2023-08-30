from PIL import Image
import numpy as np
import cv2
import struct
import os
import glob
from pathlib import Path
import shutil
last_tag = 0


def read_data_from_binary_file(filename):
    list_data = []
    with open(filename, 'rb') as f:
        f.seek(0, 0)
        first_byte = f.read(1)
        flag = struct.unpack('B', first_byte)[0]
        f.read(1)
        list_data.append(flag)
        while True:
            cur_2_bytes = f.read(2)
            if len(cur_2_bytes) < 2:
                break
            else:
                ushort_data = struct.unpack('H'*1, cur_2_bytes)[0]
                list_data.append(ushort_data >> 2)
    return list_data, flag

def decode_raw_dataset():
    # path = '/home/tao/workspace/FaceDatas/Faces0305/raw/01'
    path = '/home/tao/workspace/FaceDatas/temp_data/mouth_raw0601'
    path = path + '/*/*/'
    bins_path = sorted(glob.glob(os.path.join(path, '*.bin')))

    binsNum = len(bins_path)
    print(binsNum)

    # path = '/home/tao/workspace/YZhang/3DFace/Datas/FaceID/20201202_204058_762/'
    # names = os.listdir(path)
    # names.sort()

    # outPath = '/home/tao/workspace/FaceDatas/Faces0305/decode'
    outPath = '/home/tao/workspace/FaceDatas/temp_data/mouth_de0601'
    if not os.path.exists(outPath):  # Create the directory if not exist
        os.makedirs(outPath)

    for name in bins_path:
        if '.bin' in name:
            data, flag = read_data_from_binary_file(name)
            binName = (name.split('.')[-2]).split('/')[-1]
            className = (name.split('.')[-2]).split('/')[-3]

            if not os.path.exists(os.path.join(outPath, className)):  # Create the directory if not exist
                os.makedirs(os.path.join(outPath, className))
            if flag == 0:
                outName = outPath + '/' + className + '/' + binName + '_0.png'
                print(outName, end='')
                print('是泛光图')
            elif flag == 1:
                outName = outPath + '/' + className + '/' + binName + '_1.png'
                print(outName, end='')
                print('是散斑图')

            else:
                outName = outPath + '/' + className + '/' + binName + '_2.png'
                print(outName, end='')
                print('不确定')

            cur_img = np.reshape(np.uint8(data), (1280, 960))
            cv2.imwrite(outName, cur_img)
            # if flag == last_tag:
            #     print(name, end='')
            #     if flag == 0:
            #         print('是泛光图')
            #         outName = outPath + binName + '_IR.jpg'
            #     else:
            #         print('是散斑图')
            #         outName = outPath + binName + '_speckle.jpg'
            #     cur_img = np.reshape(np.uint8(data), (1280, 960))
            #     cv2.imwrite(outName , cur_img)
            #
            #     # cur_img = cv2.resize(cur_img, (300, 400))
            #     # cv2.imshow('', cur_img)
            #     # cv2.waitKey(1000)
            # last_tag = flag

    # data, flag = read_data_from_binary_file('20201204_041312_834.bin')
    # cur_img = np.reshape(np.uint8(data), (1280, 960))
    # cv2.imwrite('result.jpg', cur_img)
    # print('flag = %d'%flag)


def decode_one_image():
    input_path = '/media/tao/383e83e0-0daa-481c-b020-a77f6dd701a6/3DFaceDatas/temp/sh/3.raw'
    output_path = '/media/tao/383e83e0-0daa-481c-b020-a77f6dd701a6/3DFaceDatas/temp/sh/3.png'
    data, flag = read_data_from_binary_file(input_path)
    print(flag)
    cur_img = np.reshape(np.uint8(data), (1280, 1080))
    cv2.imwrite(output_path , cur_img)


def decode_shared_dataset():
    path = '/media/tao/383e83e0-0daa-481c-b020-a77f6dd701a6/3DFaceDatas/SHDataShare'
    # '/media/tao/383e83e0-0daa-481c-b020-a77f6dd701a6/3DFaceDatas/SHDataShare/FSZ0014/real/sn/SP/0.raw'
    path = path + '/*/*/*' +'/SP/'
    bins_path = sorted(glob.glob(os.path.join(path, '*.raw')))

    binsNum = len(bins_path)
    print(binsNum)

    for name in bins_path:
        if '.raw' in name:
            data, flag = read_data_from_binary_file(name)
            binName = (name.split('.')[-2]).split('/')[-1]
            temp_path, temp_file = os.path.split(name)
            pth = Path(temp_path)
            mpath = pth.parent
            opath = str(mpath) +'/Speckle/'

            if not os.path.exists(os.path.dirname(opath)):
                os.makedirs(os.path.dirname(opath))

            if flag == 0:
                # outFileName =  binName + '_0.jpg'
                outFileName =  binName + '_.jpg'
                outName = os.path.join(opath,outFileName)
                print(outName, end='')
                print('是泛光图')
            elif flag == 1:
                # outFileName = binName + '_1.jpg'
                outFileName =  binName + '.jpg'
                outName = os.path.join(opath, outFileName)
                print(outName, end='')
                print('是散斑图')

            else:
                # outFileName =  binName + '_2.jpg'
                outFileName =  binName + '_.jpg'
                outName = os.path.join(opath, outFileName)
                print(outName, end='')
                print('不确定')

            cur_img = np.reshape(np.uint8(data), (1280, 1080))
            cv2.imwrite(outName, cur_img)

def rename_shared_dataset():
    path = '/media/tao/383e83e0-0daa-481c-b020-a77f6dd701a6/3DFaceDatas/SHDataShare'
    # '/media/tao/383e83e0-0daa-481c-b020-a77f6dd701a6/3DFaceDatas/SHDataShare/FSZ0014/real/sn/SP/0.raw'
    path_person = path + '/*/'
    persons_list = sorted(glob.glob(path_person))
    personNum = len(persons_list)
    print(personNum)


    for p_index,p_name in enumerate(persons_list):
        print(p_name.split('/')[-2])

        ir_mode = p_name + '*/*' +'/IR/'+'*.jpg'
        ir_list = sorted(glob.glob(ir_mode))
        ir_imgs_num = len(ir_list)
        print('ir_imgs_num = {}'.format(ir_imgs_num))
        sp_mode = p_name + '*/*' +'/Speckle/'+'*.jpg'
        sp_list = sorted(glob.glob(sp_mode))
        sp_imgs_num = len(sp_list)
        print('sp_imgs_num = {}'.format(sp_imgs_num))

        person_imgs_count = 0

        for ir_index, ir_pathname in enumerate(ir_list):
            ir_imgName = (ir_pathname.split('.')[-2]).split('/')[-1]
            ir_tempname = ir_pathname.split('/')[-3]
            for sp_index, sp_pathname in enumerate(sp_list):
                sp_imgName = (sp_pathname.split('.')[-2]).split('/')[-1]
                sp_tempname = sp_pathname.split('/')[-3]
                if (ir_imgName == sp_imgName)&(ir_tempname == sp_tempname):
                    des_irpath = p_name + p_name.split('/')[-2] +'_'+str(person_imgs_count)+'_0.jpg'
                    des_sppath = p_name + p_name.split('/')[-2]+'_'+str(person_imgs_count)+'_1.jpg'
                    person_imgs_count = person_imgs_count + 1
                    print(ir_pathname)
                    print(des_irpath)
                    print(sp_pathname)
                    print(des_sppath)
                    shutil.copy(ir_pathname, des_irpath)
                    shutil.copy(sp_pathname, des_sppath)
                    break
                else:
                    continue

def read_binary_file(filename,is_src):
    list_data = []
    with open(filename, 'rb') as f:
        f.seek(0, 0)
        # first_byte = f.read(1)
        # flag = struct.unpack('B', first_byte)[0]
        # f.read(1)
        # list_data.append(flag)
        if is_src == 0:
            num = 224*224
        else:
            num = 1280*768

        count = 0
        while True:
            if count < num:
                cur_2_bytes = f.read(1)
                # print(cur_2_bytes)
                uint8_data = struct.unpack('B', cur_2_bytes)[0]
                list_data.append(uint8_data)
                count = count + 1
            else:
                break

    return list_data

def decode_debug_image():
    input_path = '/home/data03_disk/xianImages/test/xiaoming-00000222-ir.raw'
    output_path = '/home/data03_disk/xianImages/test/xiaoming-00000222-ir.png'
    is_src = 1
    data= read_binary_file(input_path, is_src)

    if is_src==0:
        cur_img = np.reshape(np.uint8(data), (224, 224))
    else:
        cur_img = np.reshape(np.uint8(data), (1280, 768))

    # cv2.imshow("img",cur_img)
    # cv2.waitKey()
    cv2.imwrite(output_path , cur_img)


def read_ir_binary_file(filename):
    list_data = []
    with open(filename, 'rb') as f:
        f.seek(0, 0)

        # num = 1280*768
        num = 1024*768


        count = 0
        while True:
            if count < num:
                cur_2_bytes = f.read(1)
                # print(cur_2_bytes)
                uint8_data = struct.unpack('B', cur_2_bytes)[0]
                list_data.append(uint8_data)
                count = count + 1
            else:
                break

    return list_data

def read_depth_binary_file(filename):
    list_data = []
    with open(filename, 'rb') as f:
        f.seek(0, 0)

        num = 1280*768

        count = 0
        while True:
            if count < num:
                cur_2_bytes = f.read(2)
                # print(cur_2_bytes)
                ushort_data = struct.unpack('H'*1, cur_2_bytes)[0]
                list_data.append(ushort_data)
                count = count + 1
            else:
                break

    return list_data

def decode_wq_raw_dataset():
    path = '/home/data03_disk/xianImages/12333'
    out_dir = '/home/data03_disk/xianImages/12333_decode'
    path_person = path
    persons_list = sorted(glob.glob(os.path.join(path_person, '*.raw')))
    personNum = len(persons_list)
    print(personNum)

    if not os.path.exists(out_dir):  # Create the directory if not exist
        os.makedirs(out_dir)

    for p_index, p_name in enumerate(persons_list):
        if '.raw' in p_name:
            if '-ir' in p_name:
                data = read_ir_binary_file(p_name)
                cur_img = np.reshape(np.uint8(data), (1280, 768))
                output_temp = p_name.replace('raw','png')
                output_path = output_temp.replace(path_person,out_dir)
                cv2.imwrite(output_path, cur_img)
            elif '-depth' in p_name:
                img = np.fromfile(p_name, dtype=np.uint16)
                img.shape = (1280, 768)
                img = img / 8
                cur_img = img.astype(np.uint8)

                output_temp = p_name.replace('raw', 'png')
                output_path = output_temp.replace(path_person, out_dir)
                cv2.imwrite(output_path, cur_img)

def decode_wq_bin_dataset(in_dir, out_dir):
    # path = in_dir + '/*/'
    path = in_dir

    path_person = path
    persons_list = sorted(glob.glob(os.path.join(path_person, '*.pcm')))
    personNum = len(persons_list)
    print(personNum)
    if not os.path.exists(out_dir):  # Create the directory if not exist
        os.makedirs(out_dir)

    for p_index, p_name in enumerate(persons_list):
        if '.pcm' in p_name:
            if '_b' in p_name:
                data = read_ir_binary_file(p_name)
                # cur_img = np.reshape(np.uint8(data), (1280, 768))
                cur_img = np.reshape(np.uint8(data), (1024, 768))

                output_temp = p_name.replace('pcm','png')
                output_path = output_temp.replace(in_dir,out_dir)
                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path))
                cv2.imwrite(output_path, cur_img)
            elif '_a' in p_name:
                img = np.fromfile(p_name, dtype=np.uint16)
                # img.shape = (1280, 768)
                img.shape = (1024, 768)

                img = img / 8
                cur_img = img.astype(np.uint8)
                output_temp = p_name.replace('pcm', 'png')
                output_path = output_temp.replace(in_dir, out_dir)
                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path))
                cv2.imwrite(output_path, cur_img)

def single_image_contrast(image_path,inDir, saveDir):

    image_raw = Image.open(image_path)
    image = np.array(image_raw)
    meanValue = np.mean(image)
    image1 = image.copy().astype(np.float)
    image1 = image1 * 1.35 + meanValue * (1 - 1.35)
    image1[image1 > 255] = 255
    image1 = image1.astype(np.uint8)

    output_path= image_path.replace(inDir,saveDir)
    print(output_path)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    cur_img = np.reshape(np.uint8(image1), (1280, 768))
    cv2.imwrite(output_path, cur_img)
    # image1.save(output_path)

def irImage_contrast(input_path, output_path):

    num = 0
    for path,dirs,files in os.walk(input_path):
        # img_files = list(filter(lambda x: x.endswith(('_0.png','_0.jpg')), files))
        img_files = list(filter(lambda x: x.endswith(('.png','.jpg')), files))
        if img_files != []:
            img_paths = [os.path.join(path, bin_file) for bin_file in img_files]
            for image_path in img_paths:
                print(image_path)
                num +=1
                single_image_contrast(image_path,input_path, output_path)

if __name__=='__main__':
    ## 解码wuqi的数据
    # decode_wq_raw_dataset()
    in_dir = '/home/data03_disk/bctcDebug/rawdatas0901'
    out_dir = in_dir + '_png'
    decode_wq_bin_dataset(in_dir,out_dir)
    # in_dir = '/home/data03_disk/sproFake/fake0803_png'
    # out_dir = in_dir + '_contrast'
    # irImage_contrast(in_dir, out_dir)



