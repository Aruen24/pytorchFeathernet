import os
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils import data
from torchvision import transforms as trans


def load_all_txts(txt_dir):
    files = os.listdir(txt_dir)
    txts = list(filter(lambda x: x.endswith(('txt')), files))
    txts.sort()
    txts = [os.path.join(txt_dir, txt) for txt in txts]
    total_imgs_inf = []
    for txt in txts:
        img_paths = load_data(txt)
        total_imgs_inf.extend(img_paths)
    return total_imgs_inf


def load_data(txt):
    imgs_inf = []
    f = open(txt, 'r')
    paths = f.readlines()
    paths.sort()
    for path in paths:
        path = path.strip()
        if path != '':
            imgs_inf.append(path)
    return imgs_inf


def decode_img_data(imgs_inf):
    img_paths = []
    labels = []
    for img_inf in imgs_inf:
        img_paths.append(img_inf.split('\t')[0])
        labels.append(int(img_inf.split('\t')[1]))
    data = np.transpose(np.vstack((img_paths, labels)))
    return data


def statistical_TF_sample(image_train):
    total = len(image_train)
    positive = 0
    negative = 0
    Print = 0
    Video = 0
    false_key = ['spoof', 'false', 'print', 'video']
    true_key = ['true', 'live', 'real']
    for img_path in image_train:
        img_save_dir = os.path.split(img_path)[0]
        if any(key in img_save_dir for key in false_key):
            negative += 1
            if 'print' in img_save_dir:
                Print += 1
            elif 'video' in img_save_dir:
                Video += 1
            else:
                print(img_path)
        elif any(key in img_save_dir for key in true_key):
            positive += 1
        else:
            print(img_path)
    return total, positive, negative, Print, Video


def make_data(scence_path, face_path):
    scence_data = load_all_txts(scence_path)
    face_data = np.array(load_all_txts(face_path)).reshape(-1, 1)
    scence_data = decode_img_data(scence_data)
    if len(scence_data) != len(face_data):
        print("scence img don't equal face img")
        return None
    data = np.hstack((scence_data[:, 0].reshape(-1, 1), face_data, scence_data[:, 1].reshape(-1, 1)))
    return data


def nomalization(img):
    img = np.array(img)
    img = (img - np.mean(img)) / np.std(img)
    return img


class get_data(data.Dataset):
    def __init__(self, scence_path, face_path, phase, transform):
        self.data = make_data(scence_path, face_path)
        self.phase = phase
        if self.phase == 'train':
            print('train imgs has:{}'.format(len(self.data)))
        else:
            print('test imgs has:{}'.format(len(self.data)))
        self.total, self.positive, self.negative, self.Print, self.Video = statistical_TF_sample(self.data[:, 0])
        print('imgs:%d --> [positive:%d\tnegative:%d]' % (self.total, self.positive, self.negative))
        print('In negative print:%d,video:%d' % (self.Print, self.Video))
        self.transform = transform

    def __getitem__(self, index):
        scence_imgpath = self.data[index][0]
        face_imgpath = self.data[index][1]
        label = self.data[index][2]
        img_scence = Image.open(scence_imgpath).convert('RGB')
        img_face = Image.open(face_imgpath).convert('RGB')
        if self.phase == 'train':
            img_face = self.transform(img_face)
            # img_scence = self.transform(img_scence)
        img_face = nomalization(img_face)
        img_scence = nomalization(img_scence)
        img = np.concatenate((img_scence, img_face), 2)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img.float()
        label = int(label)
        label = torch.from_numpy(np.array(label))
        return img, label

    def __len__(self):
        return len(self.data)


class get_3d_liveness_data(data.Dataset):
    def __init__(self, data_path, flag, isnorm):
        self.data = self.make_data(data_path)
        if flag == 'test':
            if isnorm:
                self.trans = trans.Compose([
                    trans.Grayscale(num_output_channels=1),
                    trans.ToTensor(),
                    trans.Normalize([0.5], [0.5]),
                ])
            else:
                self.trans = trans.Compose([
                    trans.Grayscale(num_output_channels=1),
                    trans.ToTensor(),
                    # trans.Normalize([0.5], [0.5]),
                ])
        else:
            if isnorm:
                self.trans = trans.Compose([
                    trans.Grayscale(num_output_channels=1),
                    trans.RandomResizedCrop((224, 224), scale=(0.92, 1), ratio=(1, 1)),
                    trans.RandomHorizontalFlip(),
                    trans.ColorJitter(brightness=0.5),
                    trans.ToTensor(),
                    trans.Normalize([0.5], [0.5]),
                ])
            else:
                self.trans = trans.Compose([
                    trans.Grayscale(num_output_channels=1),
                    trans.RandomResizedCrop((224, 224), scale=(0.92, 1), ratio=(1, 1)),
                    trans.RandomHorizontalFlip(),
                    trans.ColorJitter(brightness=0.5),
                    trans.ToTensor(),
                    # trans.Normalize([0.5], [0.5]),
                ])

    def __getitem__(self, index):
        img_path = self.data[index][0]
        label = self.data[index][1]
        #img = Image.open(img_path)
        #img = self.trans(img)
        #img = 255 * img
        
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (224, 224))
        img = Image.fromarray(img)
        img=self.trans(img)
        
        return img, label

    def make_data(self, data_path):
        data_lines = []
        negs = 0
        posts = 0
        for path, dirs, files in os.walk(data_path):
            imgs = list(filter(lambda x: x.endswith(('png', 'jpg')), files))
            if imgs != []:
                img_paths = [os.path.join(path, img) for img in imgs]
                for img_path in img_paths:
                    # print(img_path)
                    label = 0
                    if 'real' in img_path:
                        label = 1
                        posts += 1
                    elif 'fake' in img_path:
                        label = 0
                        negs += 1
                    data_lines.append((img_path, label))
        print('total sample is %d, postive sample is %d, negative sample is %d' % ((posts + negs), posts, negs))
        return data_lines

    def __len__(self):
        return len(self.data)


class get_3d_liveness_data_imgpath(data.Dataset):
    def __init__(self, data_path, flag, isnorm):
        self.data = self.make_data(data_path)
        if flag == 'test':
            if isnorm:
                self.trans = trans.Compose([
                    trans.Grayscale(num_output_channels=1),
                    trans.ToTensor(),
                    trans.Normalize([0.5], [0.5]),
                ])
            else:
                self.trans = trans.Compose([
                    trans.Grayscale(num_output_channels=1),
                    trans.ToTensor(),
                    # trans.Normalize([0.5], [0.5]),
                ])
        else:
            if isnorm:
                self.trans = trans.Compose([
                    trans.Grayscale(num_output_channels=1),
                    trans.RandomResizedCrop((224, 224), scale=(0.92, 1), ratio=(1, 1)),
                    trans.RandomHorizontalFlip(),
                    trans.ColorJitter(brightness=0.5),
                    trans.ToTensor(),
                    trans.Normalize([0.5], [0.5]),
                ])
            else:
                self.trans = trans.Compose([
                    trans.Grayscale(num_output_channels=1),
                    trans.RandomResizedCrop((224, 224), scale=(0.92, 1), ratio=(1, 1)),
                    trans.RandomHorizontalFlip(),
                    trans.ColorJitter(brightness=0.5),
                    trans.ToTensor(),
                    # trans.Normalize([0.5], [0.5]),
                ])

    def __getitem__(self, index):
        img_path = self.data[index][0]
        label = self.data[index][1]
        # img=Image.open(img_path)
        # img=self.trans(img)
        # img = 255 * img
        return img_path, label

    def make_data(self, data_path):
        data_lines = []
        negs = 0
        posts = 0
        for path, dirs, files in os.walk(data_path):
            imgs = list(filter(lambda x: x.endswith(('png', 'jpg')), files))
            if imgs != []:
                img_paths = [os.path.join(path, img) for img in imgs]
                for img_path in img_paths:
                    # print(img_path)
                    label = 0
                    if 'real' in img_path:
                        label = 1
                        posts += 1
                    elif 'fake' in img_path:
                        label = 0
                        negs += 1
                    data_lines.append((img_path, label))
        print('total sample is %d, postive sample is %d, negative sample is %d' % ((posts + negs), posts, negs))
        return data_lines

    def __len__(self):
        return len(self.data)











