# pytorchFeathernet多模态活体检测 
## train
```shell
#FeatherNet_m.py中修改block 中Relu6为h-swish 84、95、101行
#FeatherNetB中将SE换成CBAM

#train 修改trian_liveness.py中8、153行看是否用修改后还是修改前feathernet网络
#import FeatherNet
#import FeatherNet_m
#CUDA_VISIBLE_DEVICES=3
python train_liveness.py
python train_liveness_cbam.py


#pytorch 1.7.1版本训练报异常，降到1.6.0训练正常
```

## test
```shell
python evaluate_3dliveness_datasets.py
python evaluate_3dliveness_cbam_datasets.py
```
