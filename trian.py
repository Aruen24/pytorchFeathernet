import os
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch
from torch import nn
from torch.utils.data import DataLoader
#from pytorch import dataset, FeatherNet
import time
from datetime import datetime
from sklearn.metrics import confusion_matrix

import dataset
import FeatherNet


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Focal_loss(nn.Module):
    def __init__(self,gamm=2,eps=1e-7):
        super(Focal_loss,self).__init__()
        self.gamm=gamm
        self.eps=eps
        self.function=nn.CrossEntropyLoss(reduction='none')
    def forward(self, input,target):
        log_p=self.function(input,target)
        p=torch.exp(-log_p)
        loss=(1-p)**self.gamm*log_p
        return loss.mean()

def adjust_leanrate(optim,learn):
    for param_group in optim.param_groups:
        param_group['lr']=learn

def save_checkpoint(state,file_path):
    print('Saving variables')
    torch.save(state,file_path)

def estimate(predictions,labels):
    y_true=labels
    y_pred=predictions
    Mat_result=confusion_matrix(y_true,y_pred,labels=[0,1,2])
    sample_num=len(labels)
    acc=np.trace(Mat_result)/sample_num
    FAR = sum(iter(Mat_result[1:,0].reshape(-1, ))) / (sum(iter(Mat_result[1:, :].reshape(-1, ))) + 1e-5)
    FRR=sum(iter(Mat_result[0][1:3]))/sum(iter(Mat_result[0][:]))
    Act_acc=(np.trace(Mat_result)+Mat_result[1][2]+Mat_result[2][1])/sample_num
    result=np.hstack((FAR,FRR,acc,Act_acc))
    return result

def train(net, train_loader, optimizer, epoch,focal_loss,save_model):
    data_time = AverageMeter()
    Loss_statis=AverageMeter()
    batch_time=AverageMeter()

    net.train()
    Time=time.time()
    for i, data in enumerate(train_loader):
        input, label = data
        input = input.cuda(device=device)
        label = label.cuda(device=device)
        data_time.update(time.time() - Time)
        logits,preditions = net(input)
        loss = focal_loss(logits, label)
        Loss_statis.update(loss)
        _, preds_index = preditions.topk(1)
        preds_index=preds_index.view(-1, )
        preds_index=preds_index.detach().cpu()
        label = label.detach().cpu()
        result=estimate(preds_index,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - Time)
        Time = time.time()
        lr = optimizer.param_groups[0]['lr']
        if i % 10 == 0:
            line = 'Epoch: [{0}][{1}/{2}]\t lr: {3:.6f}\t' \
                   'load data: {data_time.val:.3f} ({data_time.avg:.3f})\t'\
                   'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                   'Loss: {loss.val:.5f} ({loss.avg:.5f})\t' \
                   'Prec: {4:.5f}' \
                .format(epoch, i, len(train_loader), lr,result[2],data_time=data_time,batch_time=batch_time, loss=Loss_statis,)
            print(line)
        if epoch==1 and i == 100:
            save_checkpoint({'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()},save_model)

def test(net,test_loader,epoch,model_dir):
   print('--------test the model--------')
   total_predictions=[]
   total_labels=[]
   net.eval()
   for i,data in enumerate(test_loader):
       input, label = data
       input=input.to(device)
       logits,predictions=net(input)
       _, preds_index = predictions.topk(1)
       preds_index = preds_index.view(-1,)
       preds_index=preds_index.detach().cpu()
       total_predictions.extend(preds_index)
       total_labels.extend(label)
   result = estimate(total_predictions, total_labels)
   print('Epoch:{}\nFalse Rejection Rate (FRR) is {:.5f}'.format(epoch,result[1]))  # 拒认率
   print('False Acceptance Rate(FAR) is {:.5f}'.format(result[0]))  # 误识率
   print("test_acc:{:.5f}".format(result[2]))
   print('Entire_Acc rata {:.5f}'.format(result[3]))
   record = open(model_dir + '/record.txt', 'a')
   record.write('Epoch:{0}\ttest_acc:{1:.5f}\tfinal_Acc:{2:.5f}\tFRR:{3:.5f}\tFAR:{4:.5f}\n'.format(epoch,result[2],result[3],result[1],result[0]))
   record.close()

def main(args):
    global device
    subdir = datetime.strftime(datetime.now(), '%m%d_%H%M')
    mode_base_dir = args.Save_model
    model_dir = os.path.join(os.path.expanduser(mode_base_dir), subdir)
    if args.Save_flag:
        if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
            os.makedirs(model_dir)

    if torch.cuda.is_available():
       device = torch.device('cuda:1')
    net= FeatherNet.FeatherNet(num_class=args.class_num, input_size=args.img_size, input_channels=args.input_channels, se=True, avgdown=True)
    # import jimukeji_net
    # net =jimukeji_net.Net(num_class=3,input_channels=6)
    #net = nn.DataParallel(net, device_ids=[0,1])
    net = net.to(device)
    print("start load train data")
    #img_transform=transforms.Compose([transforms.RandomResizedCrop((224, 224), scale=(0.9, 1), ratio=(1, 1)),transforms.ToTensor()])
    img_transform =transforms.RandomResizedCrop((224, 224), scale=(0.95, 1), ratio=(1, 1))
    img_train= dataset.get_data(args.train_scence, args.train_face, 'train', img_transform)
    img_test= dataset.get_data(args.test_scence, args.test_face, 'test', img_transform)
    train_loader = DataLoader(img_train, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=False)
    test_loader=DataLoader(img_test, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=False)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.learn_rate)
    #optimizer=torch.optim.RMSprop(params=net.parameters(),lr=args.learn_rate, momentum=0.9,alpha=0.9)
    focal_loss = Focal_loss()

    start_epoch = 1
    if args.retrain:
        print('load the model ......')
        checkpoint=torch.load(args.model_path)
        start_epoch=checkpoint['epoch']+1
        optimizer.load_state_dict(checkpoint['optimizer'])
        net.load_state_dict(checkpoint['state_dict'])

    for epoch in range(start_epoch,args.epoch_num+1):
       learn_rate = args.learn_rate * (args.learn_decay**(epoch-1))
       adjust_leanrate(optimizer, learn_rate)
       save_model = model_dir + '/' + 'Feathernet_' + str(epoch) + '.pkl'
       train(net,train_loader,optimizer,epoch,focal_loss,save_model)
       test(net,test_loader,epoch,model_dir)
       save_checkpoint({ 'epoch': epoch , 'state_dict': net.state_dict(),'optimizer': optimizer.state_dict()},save_model)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-Save_flag', type=bool,default=False,help='whether to save the model and tensorboard')
    parser.add_argument('-Save_model', type=str, help='model save path.',default='/home/data01_disk/lcw/code/train_save_file/model')
    parser.add_argument('-Save_tensorboard',type=str,help='tensorboard save path.',default='/home/data01_disk/lcw/code/train_save_file/log')
    parser.add_argument('-batch_size', type=int, help='batch size for training', default=128)
    parser.add_argument('-epoch_num', type=int, help='epoch for training', default=150)
    parser.add_argument('-img_size', type=int, help='the input size', default=224)
    parser.add_argument('-learn_rate', type=float, help='learn decay rate', default=0.005)
    parser.add_argument('-learn_decay', type=float, help='learn decay rate', default=0.95)
    parser.add_argument('-class_num', type=int, help='class num', default=3)
    parser.add_argument('-input_channels', type=int, help='the input channels', default=6)
    parser.add_argument('-retrain', type=bool, help='whether to fine-turn the model', default=False)
    parser.add_argument('-flag', type=str, help='train or evaluate the model', default='evaluate')#softmax_loss,loss1
    parser.add_argument('-model_path', type=str, help='load the model path',
                        default='/home/data01_disk/lcw/code/train_save_file/model/0824_0719/Feathernet_145.pkl')

    #set the img data
    '''trian data'''
    parser.add_argument('-train_scence', type=str, help='data for training',
                        default='/home/data01_disk/lcw/code/img_data/Scence/img_path/scence/train')
    parser.add_argument('-train_face', type=str, help='data for training',
                        default='/home/data01_disk/lcw/code/img_data/Scence/img_path/face/train')
    '''test data'''
    parser.add_argument('-test_scence', type=str, help='data for testing',
                        default='/home/data01_disk/lcw/code/img_data/Scence/img_path/scence/test')
    parser.add_argument('-test_face', type=str, help='data for testing',
                        default='/home/data01_disk/lcw/code/img_data/Scence/img_path/face/test')

    argv=parser.parse_args()
    return argv

if __name__ == '__main__':
    main(parse_arguments())