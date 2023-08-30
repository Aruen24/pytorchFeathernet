import os
import sys
ndk_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ndk_dir)

from ndk.onnx_parser import onnx_parser
from ndk.optimize import add_pre_norm, merge_layers
import os
import ndk.quantize
import numpy as np
from ndk.modelpack import save_to_file
from PIL import Image
from torch.utils import data
from torchvision import transforms as trans
import random
import cv2 as cv
import torch
from ndk.quant_tools.numpy_net import run_layers
import PIL
from torchvision import transforms as trans



def get_img_path():
    total_img_paths=[]
    img_dir=r'F:\wuqi_ndk\ndk\test_img\noface'
    for path,dirs,files in os.walk(img_dir):
        img_files = list(filter(lambda x: x.endswith(('.png', '.jpg')), files))
        if img_files != []:
            img_paths = [os.path.join(path, bin_file) for bin_file in img_files]
            total_img_paths.extend(img_paths)
    print(len(total_img_paths))
    txt=open(r'F:\wuqi_ndk\ndk\test_img\noface.txt','w')
    for img_path in total_img_paths:
        img_path=img_path.replace('\\','/')
        txt.write(img_path+'\n')
    txt.close()

class random_data(data.Dataset):
     def __init__(self,data_path):
         self.lines=[]
         if data_path.endswith('.txt'):
            reader = open(data_path, 'r')
            self.lines = reader.readlines()
         if os.path.isdir(data_path):
             for path, dirs, files in os.walk(data_path):
                 files = list(filter(lambda x: x.endswith(('.png','.jpg')), files))
                 if files != []:
                     lines = [os.path.join(path, file) for file in files]
                     self.lines.extend(lines)
     def __len__(self):
         return len(self.lines)
     def __getitem__(self, index):
         img_path=self.lines[index].strip()
         img=cv.imread(img_path)
         img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

         img=np.expand_dims(img,2)
         img=img.transpose(2,0,1)
         return img,img_path

def data_generator_input(data_path,batch_size=32):
    dataset=random_data(data_path)
    #batch_iterator = data.DataLoader(dataset, batch_size, shuffle=True, num_workers=4, drop_last=False)
    batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=4, drop_last=False))
    while True:
         data_in=next(batch_iterator)
         img=data_in[0].numpy()
         yield {'input': img}
    #return batch_iterator

def add_normal():
    model_path = r'/home/data01_disk/lcw/code/train_code/liveness/FeatherNet_moduel/FeatherNet_to_liveness/pytorch/weights/model/0812_0335/spoof_3D.onnx'
    layer_list, param_dict = onnx_parser.load_from_onnx(model_path)
    weight = np.array([1 / 255 / 0.5], dtype=np.float32)
    bias = np.array([-0.5 / 0.5], dtype=np.float32)
    add_pre_norm(layer_list, param_dict, weight, bias)
    layer_list, param_dict = merge_layers(layer_list, param_dict)
    fname=r'/home/data01_disk/lcw/code/train_code/liveness/FeatherNet_moduel/FeatherNet_to_liveness/pytorch/weights/model/0812_0335/spoof_3D'
    save_to_file(layer_list=layer_list, fname_prototxt=fname, param_dict=param_dict, fname_npz=fname)

def compare_float_and_quant(model_path,data_path):
    layer_list, param_dict = ndk.modelpack.load_from_file(fname_prototxt=model_path, fname_npz=model_path)
    node = layer_list[3].top
    generator_input = data_generator_input(data_path, 32)
    save_path='/home/data01_disk/lcw/code/retinaface/weights/NO_upsample_640_widerface+sprocomm+relu6/quant-model-480-288/quant_machinecode_E(8)_iter(3390)/'
    in_names, out_names = ndk.layers.get_net_input_output(layer_list)
    """""""""""""""""""""print input and output info"""""""""""""""""""""
    for index in range(len(in_names)):
        print(in_names[index])
    for index in range(len(out_names)):
        print(out_names[index])
    stats_dict = ndk.quantize.analyze_tensor_distribution(
        layer_list=layer_list, param_dict=param_dict,
        target_tensor_list=node, bitwidth=8,
        data_generator=generator_input,
        output_dirname=save_path,
        quant=False,
        num_batch=18,
        hw_aligned=True,
        num_bin=1024,
        max_abs_val=None, log_on=True
    )
    # Result_dict = ndk.quantize.compare_float_and_quant(layer_list=layer_list,param_dict=param_dict,
    #                                                    target_tensor_list=node,bitwidth=8,
    #                                                    data_generator=generator_input,
    #                                                    output_dirname=save_path,
    #                                                    num_batch=20,hw_aligned=True,
    #                                                    num_bin=0,max_abs_val=None,log_on=True)

    print('done')



def quant_moedel():
    fname = '/home/data01_disk/lcw/code/train_code/liveness/FeatherNet_moduel/FeatherNet_to_liveness/pytorch/weights/model/0812_0335/spoof_3D'
    layer_list, param_dict=ndk.modelpack.load_from_file(fname_prototxt=fname, fname_npz=fname)
    data_path='/home/data03_disk/YZhang/irDatas/irTrain0809'
    data_generator = data_generator_input(data_path,128)
    quant_layer_list, quant_param_dict = ndk.quantize.quantize_model(layer_list=layer_list,
                                                                     param_dict=param_dict,
                                                                     bitwidth=8,
                                                                     data_generator=data_generator,
                                                                     num_step=50,
                                                                     method_dict='KL')
    fname = '/home/data01_disk/lcw/code/train_code/liveness/FeatherNet_moduel/FeatherNet_to_liveness/pytorch/weights/model/0812_0335/quant_spoof_3D'
    save_to_file(layer_list=quant_layer_list, fname_prototxt=fname, param_dict=quant_param_dict, fname_npz=fname)



def soft_max(scores):
    scores=np.exp(scores)
    sum=scores[:, 0] + scores[:, 1]
    scores[:,0]/=sum
    scores[:, 1] /= sum
    return scores

def decode_input():
    img_path='/home/data01_disk/lcw/code/retinaface/test_img/test_data/wuqi_AEC/120.png'
    img=cv.imread(img_path,0)
    img=img.reshape(-1,)
    out_path=os.path.split(img_path)[0]+'/raw_120_input.h'
    file=open(out_path,'w')
    file.write('#include "iot_io_api.h"\nuint8_t raw_input[] = {\n')
    for i in range(len(img)):
        value = hex(img[i])
        if i != len(img)-1:
            file.write(value+',\n')
        else:
            file.write(value + '\n};')
    file.close()


def img_increase():
    import PIL
    data_path = r'F:\Dataset\3D_Collect_Data\0802'
    img_lins=[]
    for path, dirs, files in os.walk(data_path):
        files = list(filter(lambda x: x.endswith(('.png', '.jpg')), files))
        if files != []:
            lines = [os.path.join(path, file) for file in files]
            img_lins.extend(lines)
    for img_line in img_lins:
        img=cv.imread(img_line)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        image=img.copy()



        meanValue = np.mean(image)
        image1=image.copy().astype(np.float)
        image1=image1*1.35+meanValue*(1-1.35)
        image1[image1 > 255] = 255
        image1 = image1.astype(np.uint8)


        save_path=r'F:\Dataset\3D_Collect_Data\contrast/'+os.path.basename(img_line)
        cv.imwrite(save_path,image1)

        # cv.namedWindow('raw_img', 0)
        # cv.imshow('raw_img', image)
        # #cv.namedWindow('contrast_img', 0)
        # # cv.imshow('contrast_img', img_cont)
        # cv.namedWindow('contrast_my', 0)
        # cv.imshow('contrast_my', image1)
        # cv.waitKey(0)

def quanmodel_test(model_path,data_path):
    quant_model_path = model_path
    quant_layer_list, quant_param_dict = ndk.modelpack.load_from_file(fname_prototxt=quant_model_path,fname_npz=quant_model_path)
    in_names, out_names = ndk.layers.get_net_input_output(quant_layer_list)

    """""""""""""""""""""print input and output info"""""""""""""""""""""
    for index in range(len(in_names)):
        print(in_names[index])
    for index in range(len(out_names)):
        print(out_names[index])

    """""""""""""""""""""collect frac and signed of input and output"""""""""""""""""""""
    in_name = in_names[0]
    in_frac = quant_param_dict[in_name + "_frac"]
    in_signed = quant_param_dict[in_name + "_signed"]
    record_float = {}
    record_signed={}
    for out_node in out_names:
        out_node_frac = out_node + '_frac'
        out_node_flag = out_node + '_signed'
        value = quant_param_dict[out_node_frac]
        if out_node_flag in quant_param_dict.keys():
            flag=quant_param_dict[out_node_flag]
        else:
            flag= True
        record_float[out_node] = value
        record_signed[out_node] = flag
    print(record_float)

    # """""""""""""""""""""generate bin file for runing on edge device"""""""""""""""""""""
    # ndk.modelpack.modelpack_from_file(
    #     bitwidth=8,
    #     fname_prototxt='./Quant_facedetect_lcw/320_192/new_trainSize_640/model/model.prototxt',
    #     fname_npz='./Quant_facedetect_lcw/320_192/new_trainSize_640/model/model.npz',
    #     out_file_path='./Quant_facedetect_lcw/320_192/new_trainSize_640/model/outbin',
    #     model_name='facedetect',
    #     use_machine_code = True
    # )
    data_generator = data_generator_input(data_path,batch_size=1)
    for i,data in enumerate(data_generator):
        img_path = data[1]
        data_in = data[0].numpy()
        logits = run_layers(input_data_batch=data_in,
                            layer_list=quant_layer_list,
                            target_feature_tensor_list=out_names,
                            param_dict=quant_param_dict,
                            quant=True)

        #np.savez(r"F:\wuqi_ndk\ndk\output\h_file/input.npz", **logits)
        # inputname=os.path.basename(img_path[0])
        # filename = '/home/data01_disk/lcw/code/retinaface/test_img/compare_pc_and_wuqi/test/' + inputname.replace('png','.h')
        # ndk.utils.save_quantized_feature_to_header(filename,
        #                                  feature=logits[in_name],#
        #                                  bit_width=8,
        #                                  frac=in_frac,
        #                                  signed=in_signed,#True
        #                                  name_in_header='resize_input',
        #                                  aligned=True)
        # ''''save quant reslut in .h file'''
        # ''''logits--[land1,class1,box1,land2,class2,box2,land3,class3,box3]''''
        # h_files=['land_0','class_0','box_0','land_1','class_1','box_1','land_2','class_2','box_2']
        # for i in range(9):
        #     filename='./output/h_file/output/'+h_files[i]+'.h'
        #     ndk.utils.save_quantized_feature_to_header(filename,
        #                                      feature=logits[out_names[i]],
        #                                      bit_width=8,
        #                                      frac=record_float[out_names[i]],
        #                                      signed=record_signed[out_names[i]],#True
        #                                      name_in_header='feature',
        #                                      aligned=True)

        boxs=[]
        classify=[]
        landmks=[]
        boxs.extend(logits['output0'].transpose(0, 2, 3, 1).squeeze(0).reshape(-1,4))
        boxs.extend(logits['output1'].transpose(0, 2, 3, 1).squeeze(0).reshape(-1,4))
        boxs.extend(logits['output2'].transpose(0, 2, 3, 1).squeeze(0).reshape(-1, 4))
        boxs = np.array(boxs)
        classify.extend(logits['448'].transpose(0, 2, 3, 1).squeeze(0).reshape(-1,2))
        classify.extend(logits['449'].transpose(0, 2, 3, 1).squeeze(0).reshape(-1, 2))
        classify.extend(logits['450'].transpose(0, 2, 3, 1).squeeze(0).reshape(-1, 2))
        classify = np.array(classify)
        landmks.extend(logits['451'].transpose(0, 2, 3, 1).squeeze(0).reshape(-1,10))
        landmks.extend(logits['452'].transpose(0, 2, 3, 1).squeeze(0).reshape(-1, 10))
        landmks.extend(logits['453'].transpose(0, 2, 3, 1).squeeze(0).reshape(-1, 10))
        landmks=np.array(landmks)
        validate_reslut(boxs,classify,landmks,480,288,img_path[0],os.path.basename(data_path),'')
        #validate_reslut(boxs, classify, landmks, 320, 192, img_path[0], os.path.basename(data_path))
        print('done')


def sort_img():
    txt_path='/home/data03_disk/lcw/sprocomm_face_data/sprocomm_guangming.txt'
    new_path='/home/data03_disk/lcw/sprocomm_face_data/sprocomm_guangming1.txt'
    file=open(txt_path,'r')
    file1=open(new_path,'w')
    lines=file.readlines()
    for line in lines:
        if '/home/' in line:
            file1.write(line)
        else:
            line=line.strip().split()
            ls=len(line)
            if ls==5:
                for i in range(ls):
                    if i <4:
                        file1.write(line[i]+' ')
                    if i==4:
                        file1.write('-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 1.0'+'\n')
            if ls==15:
                for i in range(4):
                    file1.write(line[i]+' ')
                for i in range(5):
                    file1.write(line[2*i+5] + ' '+line[2*i+6]+' 1.0 ')
                file1.write(line[4]+'\n')





if __name__=='__main__':
    model_path=r'/home/data01_disk/lcw/code/retinaface/weights/NO_upsample_640_widerface+sprocomm+relu6_add_data/quant-model-480-288/train_input_640/quant_machinecode_E(5)_iter(2500)/after_train_quant_model'
    data_path=r'/home/data01_disk/lcw/code/retinaface/test_img/111'
    #img_increase()
    #quanmodel_test(model_path,data_path)
    #compare_float_and_quant(model_path,data_path)
    #decode_input()
    #revise_relu_to_concat(model_path)
    add_normal()
    quant_moedel()
